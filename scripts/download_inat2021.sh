#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:-mini}
case "$MODE" in
    mini|full) ;;
    *)
        echo "usage: $0 {mini|full}" >&2
        exit 64
        ;;
esac

BASE=${BASE:?BASE must be set to the dataset storage root}
DOWNLOAD_PROXY=${DOWNLOAD_PROXY-}
RUN_ID=${RUN_ID:-download-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-100}
MAX_MD5_REDOWNLOADS=${MAX_MD5_REDOWNLOADS:-2}
RETRY_BASE_DELAY=${RETRY_BASE_DELAY:-5}
HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-30}
CHILD_STOP_ATTEMPTS=${CHILD_STOP_ATTEMPTS:-50}

require_uint() {
    local name=$1
    local value=$2
    local minimum=$3
    if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < minimum )); then
        echo "$name must be an integer greater than or equal to $minimum" >&2
        exit 64
    fi
}
require_uint MAX_ATTEMPTS "$MAX_ATTEMPTS" 1
require_uint MAX_MD5_REDOWNLOADS "$MAX_MD5_REDOWNLOADS" 0
require_uint RETRY_BASE_DELAY "$RETRY_BASE_DELAY" 0
require_uint HEARTBEAT_INTERVAL "$HEARTBEAT_INTERVAL" 1
require_uint CHILD_STOP_ATTEMPTS "$CHILD_STOP_ATTEMPTS" 1
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_ID must be a non-empty safe path component" >&2
    exit 64
fi

ARCHIVE=$BASE/archive
STATE_DIR=$BASE/metadata/$RUN_ID
STATE=$STATE_DIR/status
mkdir -p "$ARCHIVE" "$BASE/runs"

LOCK_PATH=$ARCHIVE/.download-inat2021.lock
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "[DOWNLOAD] lock-busy archive=$ARCHIVE" >&2
    exit 73
fi
mkdir -p "$STATE_DIR"

PROCESS_ID=$$
STARTED_AT=$(date -u +%FT%TZ)
ATTEMPT_SESSION=$(date -u +%Y%m%dT%H%M%S%NZ)-$$-$RANDOM
CURRENT_FILE=
CURRENT_ATTEMPT=0
CURRENT_PHASE=
LAST_COMPLETED_FILE=
LAST_COMPLETED_BYTES=0
LAST_COMPLETED_MD5=
FAILURE_REASON=
HEARTBEAT_PID=
ACTIVE_CHILD_PID=
ACTIVE_CHILD_IS_SUPERVISOR=0
HEARTBEAT_STOP=$STATE_DIR/.heartbeat-stop-$ATTEMPT_SESSION
CHILD_RC=0

write_state() {
    local status=$1
    local exit_code=${2:-}
    local temporary=$STATE.tmp.${BASHPID}
    local downloaded_bytes=0
    if [[ -n "$CURRENT_FILE" && -f "$ARCHIVE/$CURRENT_FILE.part" ]]; then
        downloaded_bytes=$(stat -c %s "$ARCHIVE/$CURRENT_FILE.part")
    elif [[ -n "$CURRENT_FILE" && -f "$ARCHIVE/$CURRENT_FILE" ]]; then
        downloaded_bytes=$(stat -c %s "$ARCHIVE/$CURRENT_FILE")
    fi
    {
        printf 'status=%s\n' "$status"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'mode=%s\n' "$MODE"
        printf 'pid=%s\n' "$PROCESS_ID"
        printf 'child_pid=%s\n' "$ACTIVE_CHILD_PID"
        printf 'started_at=%s\n' "$STARTED_AT"
        printf 'heartbeat_at=%s\n' "$(date -u +%FT%TZ)"
        printf 'current_file=%s\n' "$CURRENT_FILE"
        printf 'phase=%s\n' "$CURRENT_PHASE"
        printf 'attempt=%s\n' "$CURRENT_ATTEMPT"
        printf 'downloaded_bytes=%s\n' "$downloaded_bytes"
        printf 'last_completed_file=%s\n' "$LAST_COMPLETED_FILE"
        printf 'last_completed_bytes=%s\n' "$LAST_COMPLETED_BYTES"
        printf 'last_completed_md5=%s\n' "$LAST_COMPLETED_MD5"
        printf 'failure_reason=%s\n' "$FAILURE_REASON"
        if [[ -n "$exit_code" ]]; then
            printf 'exit_code=%s\n' "$exit_code"
        fi
    } > "$temporary"
    mv "$temporary" "$STATE"
    sync "$STATE"
}

stop_heartbeat() {
    if [[ -n "$HEARTBEAT_PID" ]]; then
        touch "$HEARTBEAT_STOP"
        wait "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=
        rm -f "$HEARTBEAT_STOP"
    fi
    return 0
}

heartbeat_while_running() {
    local child_pid=$1
    while kill -0 "$child_pid" 2>/dev/null && [[ ! -e "$HEARTBEAT_STOP" ]]; do
        if ! kill -0 "$PROCESS_ID" 2>/dev/null; then
            kill -TERM "$child_pid" 2>/dev/null || true
            sleep 0.1
            kill -KILL "$child_pid" 2>/dev/null || true
            return
        fi
        write_state running
        local elapsed
        for ((elapsed = 0; elapsed < HEARTBEAT_INTERVAL; elapsed++)); do
            if [[ -e "$HEARTBEAT_STOP" ]] || ! kill -0 "$child_pid" 2>/dev/null; then
                return
            fi
            if ! kill -0 "$PROCESS_ID" 2>/dev/null; then
                kill -TERM "$child_pid" 2>/dev/null || true
                sleep 0.1
                kill -KILL "$child_pid" 2>/dev/null || true
                return
            fi
            sleep 1
        done
    done
}

stop_active_child() {
    if [[ -z "$ACTIVE_CHILD_PID" ]]; then
        return 0
    fi
    kill -TERM "$ACTIVE_CHILD_PID" 2>/dev/null || true
    local stop_attempts=$CHILD_STOP_ATTEMPTS
    if (( ACTIVE_CHILD_IS_SUPERVISOR == 1 )); then
        stop_attempts=$((stop_attempts + 20))
    fi
    local waited
    for ((waited = 0; waited < stop_attempts; waited++)); do
        if ! kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
        kill -KILL "$ACTIVE_CHILD_PID" 2>/dev/null || true
    fi
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    ACTIVE_CHILD_PID=
    ACTIVE_CHILD_IS_SUPERVISOR=0
}

wait_for_active_child() {
    heartbeat_while_running "$ACTIVE_CHILD_PID" &
    HEARTBEAT_PID=$!
    set +e
    wait "$ACTIVE_CHILD_PID"
    CHILD_RC=$?
    set -e
    ACTIVE_CHILD_PID=
    stop_heartbeat
}

supervise_active_child() {
    local supervisor_signal_code=0
    trap 'rc=$?; trap - EXIT HUP INT TERM; stop_active_child; stop_heartbeat; exit "$rc"' EXIT
    trap 'if [[ -n "$ACTIVE_CHILD_PID" ]]; then exit 129; else supervisor_signal_code=129; fi' HUP
    trap 'if [[ -n "$ACTIVE_CHILD_PID" ]]; then exit 130; else supervisor_signal_code=130; fi' INT
    trap 'if [[ -n "$ACTIVE_CHILD_PID" ]]; then exit 143; else supervisor_signal_code=143; fi' TERM

    if (( supervisor_signal_code != 0 )); then
        return "$supervisor_signal_code"
    fi
    "$@" &
    ACTIVE_CHILD_PID=$!
    if (( supervisor_signal_code != 0 )); then
        return "$supervisor_signal_code"
    fi
    wait_for_active_child
    return "$CHILD_RC"
}

run_active_child() {
    local parent_signal_name=
    local parent_signal_code=0
    trap 'parent_signal_name=HUP; parent_signal_code=129' HUP
    trap 'parent_signal_name=INT; parent_signal_code=130' INT
    trap 'parent_signal_name=TERM; parent_signal_code=143' TERM

    if (( parent_signal_code != 0 )); then
        trap 'terminate HUP 129' HUP
        trap 'terminate INT 130' INT
        trap 'terminate TERM 143' TERM
        terminate "$parent_signal_name" "$parent_signal_code"
    fi
    supervise_active_child "$@" &
    ACTIVE_CHILD_PID=$!
    ACTIVE_CHILD_IS_SUPERVISOR=1
    trap 'terminate HUP 129' HUP
    trap 'terminate INT 130' INT
    trap 'terminate TERM 143' TERM
    if (( parent_signal_code != 0 )); then
        terminate "$parent_signal_name" "$parent_signal_code"
    fi

    set +e
    wait "$ACTIVE_CHILD_PID"
    CHILD_RC=$?
    set -e
    ACTIVE_CHILD_PID=
    ACTIVE_CHILD_IS_SUPERVISOR=0
    return 0
}

verify_md5() {
    local expected_md5=$1
    local path=$2
    local check_file=$STATE_DIR/md5-$ATTEMPT_SESSION-${BASHPID}.txt
    CURRENT_PHASE=md5
    write_state running
    printf '%s  %s\n' "$expected_md5" "$path" > "$check_file"
    run_active_child md5sum -c "$check_file"
    local rc=$CHILD_RC
    rm -f "$check_file"
    CURRENT_PHASE=
    return "$rc"
}

terminate() {
    local signal_name=$1
    local exit_code=$2
    FAILURE_REASON=signal-${signal_name,,}
    exit "$exit_code"
}

finish() {
    local rc=$?
    trap - EXIT
    stop_heartbeat
    stop_active_child
    if (( rc == 0 )); then
        write_state passed 0
        echo "[DOWNLOAD] status=passed run_id=$RUN_ID state=$STATE"
    else
        write_state failed "$rc"
        echo "[DOWNLOAD] status=failed exit=$rc run_id=$RUN_ID state=$STATE"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'terminate HUP 129' HUP
trap 'terminate INT 130' INT
trap 'terminate TERM 143' TERM

retry_delay() {
    local attempt=$1
    local delay=$((attempt * RETRY_BASE_DELAY))
    if (( delay > 60 )); then
        delay=60
    fi
    printf '%s' "$delay"
}

permanent_http_status() {
    local attempt_log=$1
    local line
    local status=
    while IFS= read -r line; do
        if [[ "$line" =~ HTTP/[0-9.]+[[:space:]]+(4[0-9][0-9])([[:space:]]|$) ]]; then
            status=${BASH_REMATCH[1]}
        elif [[ "$line" =~ ERROR[[:space:]]+(4[0-9][0-9])(:|[[:space:]]) ]]; then
            status=${BASH_REMATCH[1]}
        fi
    done < "$attempt_log"
    case "$status" in
        ""|408|409|416|425|429) return 1 ;;
        *) printf '%s' "$status" ;;
    esac
}
download_one() {
    local name=$1
    local url=$2
    local expected_md5=$3
    local expected_bytes=$4
    local final=$ARCHIVE/$name
    local partial=$final.part
    local md5_redownloads=0

    CURRENT_FILE=$name
    CURRENT_ATTEMPT=0
    CURRENT_PHASE=
    FAILURE_REASON=

    if [[ -f "$final" ]]; then
        if verify_md5 "$expected_md5" "$final"; then
            LAST_COMPLETED_FILE=$name
            LAST_COMPLETED_BYTES=$(stat -c %s "$final")
            LAST_COMPLETED_MD5=$expected_md5
            write_state running
            echo "[DOWNLOAD] already-complete name=$name bytes=$LAST_COMPLETED_BYTES md5=$expected_md5"
            return
        fi
        local invalid_final
        invalid_final=$final.bad.$(date -u +%Y%m%dT%H%M%SZ).$$
        mv "$final" "$invalid_final"
        sync -f "$ARCHIVE"
        echo "[DOWNLOAD] existing-final-md5-mismatch name=$name quarantined=$invalid_final"
    fi

    while (( md5_redownloads <= MAX_MD5_REDOWNLOADS )); do
        touch "$partial"
        local transfer_attempt=0
        local completed=0
        local md5_verified=0
        local restart_from_zero=0
        while (( transfer_attempt < MAX_ATTEMPTS )); do
            transfer_attempt=$((transfer_attempt + 1))
            CURRENT_ATTEMPT=$((CURRENT_ATTEMPT + 1))
            local before_bytes
            before_bytes=$(stat -c %s "$partial")
            CURRENT_PHASE=transfer
            FAILURE_REASON=
            write_state running
            echo "[DOWNLOAD] attempt=$CURRENT_ATTEMPT name=$name resume_bytes=$before_bytes"

            local attempt_log
            attempt_log=$STATE_DIR/wget-${name//[^A-Za-z0-9._-]/_}-$ATTEMPT_SESSION-$CURRENT_ATTEMPT.log
            run_active_child \
                env http_proxy="$DOWNLOAD_PROXY" https_proxy="$DOWNLOAD_PROXY" \
                wget \
                    --continue \
                    --output-document="$partial" \
                    --tries=1 \
                    --timeout=60 \
                    --dns-timeout=30 \
                    --connect-timeout=30 \
                    --read-timeout=60 \
                    --server-response \
                    --progress=dot:giga \
                    "$url" 2>"$attempt_log"
            local rc=$CHILD_RC
            CURRENT_PHASE=
            cat "$attempt_log" >&2

            local after_bytes
            after_bytes=$(stat -c %s "$partial")
            if (( rc == 0 && after_bytes < expected_bytes )); then
                FAILURE_REASON=retryable-transfer
                write_state running
                echo "[DOWNLOAD] retryable-incomplete-success name=$name expected_bytes=$expected_bytes preserved_bytes=$after_bytes"
                if (( transfer_attempt < MAX_ATTEMPTS )); then
                    sleep "$(retry_delay "$CURRENT_ATTEMPT")"
                fi
                continue
            fi
            if (( rc == 0 && after_bytes > expected_bytes )); then
                FAILURE_REASON=size-mismatch
                write_state failed 74
                echo "[DOWNLOAD] oversized-transfer name=$name expected_bytes=$expected_bytes actual_bytes=$after_bytes" >&2
                return 74
            fi
            if (( rc == 0 )); then
                echo "[DOWNLOAD] transfer-complete name=$name bytes=$after_bytes"
                completed=1
                break
            fi

            if grep -Eq 'HTTP/[0-9.]+ +416( |$)|ERROR +416(:| )' "$attempt_log"; then
                echo "[DOWNLOAD] range-not-satisfiable name=$name bytes=$after_bytes"
                if verify_md5 "$expected_md5" "$partial"; then
                    echo "[DOWNLOAD] complete-partial-verified name=$name bytes=$after_bytes"
                    completed=1
                    md5_verified=1
                    break
                fi
                local range_quarantine
                range_quarantine=$partial.bad.$(date -u +%Y%m%dT%H%M%SZ).$$.range416.$md5_redownloads
                mv "$partial" "$range_quarantine"
                sync -f "$ARCHIVE"
                FAILURE_REASON=range-416-md5-mismatch
                write_state running
                echo "[DOWNLOAD] range-416-md5-mismatch name=$name quarantined=$range_quarantine" >&2
                if (( md5_redownloads == MAX_MD5_REDOWNLOADS )); then
                    return 74
                fi
                md5_redownloads=$((md5_redownloads + 1))
                restart_from_zero=1
                break
            fi

            local permanent_status
            if permanent_status=$(permanent_http_status "$attempt_log"); then
                FAILURE_REASON=permanent-http
                write_state failed 69
                echo "[DOWNLOAD] permanent-http-failure name=$name http=$permanent_status exit=$rc preserved_bytes=$after_bytes" >&2
                return 69
            fi

            FAILURE_REASON=retryable-transfer
            write_state running
            echo "[DOWNLOAD] retryable-failure name=$name exit=$rc preserved_bytes=$after_bytes"
            if (( transfer_attempt < MAX_ATTEMPTS )); then
                sleep "$(retry_delay "$CURRENT_ATTEMPT")"
            fi
        done

        if (( restart_from_zero == 1 )); then
            continue
        fi
        if (( completed == 0 )); then
            FAILURE_REASON=transfer-attempts-exhausted
            echo "[DOWNLOAD] exhausted name=$name attempts=$CURRENT_ATTEMPT bytes=$(stat -c %s "$partial")" >&2
            return 75
        fi

        if (( md5_verified == 1 )) || verify_md5 "$expected_md5" "$partial"; then
            mv "$partial" "$final"
            sync -f "$ARCHIVE"
            LAST_COMPLETED_FILE=$name
            LAST_COMPLETED_BYTES=$(stat -c %s "$final")
            LAST_COMPLETED_MD5=$expected_md5
            FAILURE_REASON=
            write_state running
            echo "[DOWNLOAD] complete name=$name bytes=$LAST_COMPLETED_BYTES md5=$expected_md5"
            return
        fi

        local quarantined
        quarantined=$partial.bad.$(date -u +%Y%m%dT%H%M%SZ).$$.$md5_redownloads
        mv "$partial" "$quarantined"
        sync -f "$ARCHIVE"
        FAILURE_REASON=md5-mismatch
        write_state running
        echo "[DOWNLOAD] md5-mismatch name=$name quarantined=$quarantined redownloads=$md5_redownloads" >&2
        if (( md5_redownloads == MAX_MD5_REDOWNLOADS )); then
            echo "[DOWNLOAD] md5-redownloads-exhausted name=$name count=$md5_redownloads" >&2
            return 74
        fi
        md5_redownloads=$((md5_redownloads + 1))
    done
}

case "$MODE" in
    mini)
        download_one \
            train_mini.tar.gz \
            https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz \
            db6ed8330e634445efc8fec83ae81442 \
            44636137542
        download_one \
            val.tar.gz \
            https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz \
            f6f6e0e242e3d4c9569ba56400938afc \
            8931661582
        ;;
    full)
        download_one \
            train.tar.gz \
            https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz \
            e0526d53c7f7b2e3167b2b43bb2690ed \
            239909164970
        download_one \
            val.tar.gz \
            https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz \
            f6f6e0e242e3d4c9569ba56400938afc \
            8931661582
        ;;
    *)
        echo "usage: $0 {mini|full}" >&2
        exit 64
        ;;
esac

CURRENT_FILE=
CURRENT_ATTEMPT=0
FAILURE_REASON=
