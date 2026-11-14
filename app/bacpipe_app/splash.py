"""
The splash screen shown while the application starts, run as its own process:

    python splash.py path/to/splash.png

It shows the image in a small borderless window until its standard input is
closed (the application does that when the browser page is ready), or until
the timeout is reached.

The script only uses the standard library (tkinter) and must not import
bacpipe_app, so that it appears as fast as possible.
"""

import sys
import threading
import time

TIMEOUT_SECONDS = 180


def show(image_path):
    import tkinter

    root = tkinter.Tk()
    root.overrideredirect(True)  # no title bar
    root.configure(bg="white", highlightthickness=1, highlightbackground="#bbbbbb")
    try:
        root.attributes("-topmost", True)
    except tkinter.TclError:
        pass

    image = tkinter.PhotoImage(file=image_path)
    tkinter.Label(root, image=image, bg="white", borderwidth=0).pack(
        padx=30, pady=(25, 5)
    )
    tkinter.Label(
        root, text="starting …", bg="white", fg="#666666", font=("Helvetica", 11)
    ).pack(pady=(0, 18))

    # centre the window on the screen
    root.update_idletasks()
    width, height = root.winfo_reqwidth(), root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")

    done = threading.Event()

    def wait_for_parent():
        # read() returns when the application closes the pipe or exits
        try:
            sys.stdin.buffer.read()
        except Exception:
            time.sleep(TIMEOUT_SECONDS)
        done.set()

    threading.Thread(target=wait_for_parent, daemon=True).start()
    started = time.monotonic()

    def poll():
        if done.is_set() or time.monotonic() - started > TIMEOUT_SECONDS:
            root.destroy()
        else:
            root.after(100, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    try:
        show(sys.argv[1])
    except Exception as error:  # no tkinter / no display: start without splash
        sys.stderr.write(f"splash screen not available: {error}\n")
        sys.exit(1)
