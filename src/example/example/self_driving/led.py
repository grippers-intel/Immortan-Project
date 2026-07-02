import RPi.GPIO as GPIO
import threading
import time

PIN_GREEN = 24
PIN_RED = 25
PIN_YELLOW_LEFT = 23
PIN_YELLOW_RIGHT = 18
ALL_PINS = [PIN_GREEN, PIN_RED, PIN_YELLOW_LEFT, PIN_YELLOW_RIGHT]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in ALL_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

_blink_stop_event = None
_blink_thread = None
current_mode = None
mode_lock = threading.Lock()


def stop_blink():
    global _blink_stop_event, _blink_thread
    if _blink_stop_event:
        _blink_stop_event.set()   # 스레드에게 종료 신호만 보냄
    # join 없음 — daemon 스레드가 알아서 종료됨
    _blink_stop_event = None
    _blink_thread = None


def start_blink(pins, interval=0.3):
    global _blink_stop_event, _blink_thread
    stop_blink()
    ev = threading.Event()
    _blink_stop_event = ev

    def worker():
        state = 1
        while not ev.is_set():
            for p in pins:
                GPIO.output(p, state)
            state = 1 - state
            ev.wait(interval)   # set되면 즉시 깨어남

    _blink_thread = threading.Thread(target=worker, daemon=True)
    _blink_thread.start()


def set_mode(name, fn):
    global current_mode
    with mode_lock:
        if current_mode == name:
            return
        current_mode = name
        fn()


def mode_straight():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        GPIO.output(PIN_GREEN, 1)
    set_mode("straight", _do)


def mode_turn_right():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        start_blink([PIN_YELLOW_RIGHT])
    set_mode("turn_right", _do)


def mode_stop():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        GPIO.output(PIN_RED, 1)
    set_mode("stop", _do)


def mode_park_done():
    def _do():
        stop_blink()
        for p in ALL_PINS:
            GPIO.output(p, 0)
        start_blink(ALL_PINS, interval=0.3)
    set_mode("park_done", _do)


def all_off():
    global current_mode
    with mode_lock:
        current_mode = None
    stop_blink()
    for p in ALL_PINS:
        GPIO.output(p, 0)


def cleanup():
    all_off()
    GPIO.cleanup()