

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path
import cv2
import threading
import winsound
import os

from PIL import Image, ImageTk
from ultralytics import YOLO


# =====================================================
# THEME
# =====================================================

BG = "#0B1120"
SIDEBAR = "#091A32"
PANEL = "#101F38"
PANEL_LIGHT = "#172B49"
BORDER = "#24466D"

WHITE = "#F8FAFC"
MUTED = "#94A3B8"

BLUE = "#38BDF8"
PURPLE = "#A855F7"
GREEN = "#00E69A"
RED = "#F43F5E"
ORANGE = "#F59E0B"


class HumanMotionDetectionGUI:

    def __init__(self, root):

        self.root = root

        self.root.title("AI Human Motion Detection System")
        self.root.geometry("1536x950")
        self.root.minsize(1100, 720)
        self.root.configure(bg=BG)

        # CAMERA
        self.camera = None
        self.model = None
        self.camera_running = False
        self.camera_thread = None

        self.stop_event = threading.Event()
        self.shutdown_event = threading.Event()

        # ALARM
        self.alarm_event = threading.Event()

        self.alarm_thread = threading.Thread(
            target=self.alarm_loop,
            daemon=True
        )
        self.alarm_thread.start()

        # RECORDING
        self.video_writer = None
        self.recording_active = False
        self.recording_start_time = None
        self.recording_path = None

        # FOLDERS
        self.project_folder = Path(__file__).resolve().parent
        self.recording_folder = self.project_folder / "recordings"
        self.recording_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.history_counter = 0

        # ANIMATION
        self.animation_phase = 0
        self.animation_job = None
        self.animation_colors = [
            BLUE,
            "#60A5FA",
            PURPLE,
            BLUE
        ]

        # GUI
        self.setup_styles()
        self.create_dashboard()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

        # Start GUI animations
        self.animate_dashboard()

    # =================================================
    # STYLES
    # =================================================

    def setup_styles(self):

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Treeview",
            background=PANEL,
            foreground=WHITE,
            fieldbackground=PANEL,
            rowheight=36,
            borderwidth=0,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Treeview.Heading",
            background=PANEL_LIGHT,
            foreground=BLUE,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padding=10
        )

        style.map(
            "Treeview",
            background=[("selected", "#263D60")],
            foreground=[("selected", WHITE)]
        )

    # =================================================
    # DASHBOARD
    # =================================================

    def create_dashboard(self):

        self.container = tk.Frame(
            self.root,
            bg=BG
        )

        self.container.pack(
            fill="both",
            expand=True
        )

        self.create_sidebar()

        self.main_area = tk.Frame(
            self.container,
            bg=BG
        )

        self.main_area.pack(
            side="right",
            fill="both",
            expand=True
        )

        self.create_header()
        self.create_status_section()
        self.create_content_section()
        self.create_history_section()

    # =================================================
    # ANIMATED BUTTON
    # =================================================

    def create_button(
        self,
        parent,
        text,
        color,
        command,
        width=16
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 11, "bold"),
            bg=color,
            fg=WHITE,
            activebackground=color,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            width=width,
            pady=13,
            cursor="hand2"
        )

        hover_color = {
            "#168B52": "#20B96D",
            "#B93838": "#F04452"
        }.get(color, PANEL_LIGHT)

        button.bind(
            "<Enter>",
            lambda event: self.button_hover(
                button, hover_color
            )
        )

        button.bind(
            "<Leave>",
            lambda event: self.button_hover(
                button, color
            )
        )

        return button

    def button_hover(self, button, color):

        if str(button["state"]) != "disabled":
            button.config(bg=color)

    # =================================================
    # SIDEBAR
    # =================================================

    def create_sidebar(self):

        self.sidebar = tk.Frame(
            self.container,
            bg=SIDEBAR,
            width=265,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        self.sidebar.pack_propagate(False)

        # LOGO
        logo = tk.Frame(
            self.sidebar,
            bg=SIDEBAR
        )

        logo.pack(
            fill="x",
            padx=25,
            pady=(30, 22)
        )

        self.logo_icon = tk.Label(
            logo,
            text="◈",
            font=("Segoe UI", 43, "bold"),
            fg=BLUE,
            bg=SIDEBAR
        )

        self.logo_icon.pack(anchor="w")

        tk.Label(
            logo,
            text="AI SECURITY",
            font=("Segoe UI", 22, "bold"),
            fg=WHITE,
            bg=SIDEBAR
        ).pack(anchor="w")

        tk.Label(
            logo,
            text="HUMAN MOTION DETECTOR",
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=SIDEBAR
        ).pack(anchor="w", pady=(6, 0))

        # START CAMERA
        self.start_button = self.create_button(
            self.sidebar,
            "▶  START CAMERA",
            "#168B52",
            self.start_camera,
            19
        )

        self.start_button.pack(
            fill="x",
            padx=18,
            pady=(12, 7)
        )

        # STOP CAMERA
        self.stop_button = self.create_button(
            self.sidebar,
            "■  STOP CAMERA",
            "#B93838",
            self.stop_camera,
            19
        )

        self.stop_button.pack(
            fill="x",
            padx=18,
            pady=(5, 20)
        )

        # Always visible
        self.stop_button.config(
            state="disabled"
        )

        tk.Frame(
            self.sidebar,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=18,
            pady=5
        )

        # MENU
        tk.Label(
            self.sidebar,
            text="MAIN MENU",
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=SIDEBAR
        ).pack(
            anchor="w",
            padx=25,
            pady=(17, 12)
        )

        menu_items = [
            ("▦", "Dashboard", self.go_dashboard),
            ("◉", "Live Camera", self.go_camera),
            ("▤", "Detection History", self.go_history),
            ("▣", "Recordings", self.open_recordings),
            ("⚙", "Settings", self.show_settings)
        ]

        self.menu_buttons = []

        for icon, title, command in menu_items:

            button = self.create_sidebar_button(
                icon,
                title,
                command
            )

            self.menu_buttons.append(button)

        # SYSTEM STATUS
        bottom = tk.Frame(
            self.sidebar,
            bg=SIDEBAR
        )

        bottom.pack(
            side="bottom",
            fill="x",
            padx=20,
            pady=25
        )

        tk.Frame(
            bottom,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            pady=(0, 18)
        )

        tk.Label(
            bottom,
            text="SYSTEM STATUS",
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=SIDEBAR
        ).pack(anchor="w")

        self.sidebar_status = tk.Label(
            bottom,
            text="● AI Detection Ready",
            font=("Segoe UI", 10, "bold"),
            fg=GREEN,
            bg=SIDEBAR
        )

        self.sidebar_status.pack(
            anchor="w",
            pady=12
        )

        tk.Label(
            bottom,
            text="Python | YOLO | OpenCV",
            font=("Segoe UI", 9),
            fg="#64748B",
            bg=SIDEBAR
        ).pack(anchor="w")

    def create_sidebar_button(
        self,
        icon,
        title,
        command
    ):

        button = tk.Button(
            self.sidebar,
            text=f"   {icon}     {title}",
            font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR,
            fg="#CBD5E1",
            activebackground=PANEL_LIGHT,
            activeforeground=BLUE,
            relief="flat",
            bd=0,
            anchor="w",
            padx=18,
            pady=15,
            cursor="hand2",
            command=command
        )

        button.pack(
            fill="x",
            padx=12,
            pady=3
        )

        button.bind(
            "<Enter>",
            lambda event: button.config(
                bg=PANEL_LIGHT
            )
        )

        button.bind(
            "<Leave>",
            lambda event: button.config(
                bg=SIDEBAR
            )
        )

        return button

    # =================================================
    # HEADER
    # =================================================

    def create_header(self):

        header = tk.Frame(
            self.main_area,
            bg=BG
        )

        header.pack(
            fill="x",
            padx=25,
            pady=(22, 17)
        )

        left = tk.Frame(header, bg=BG)
        left.pack(side="left")

        tk.Label(
            left,
            text="Security Dashboard",
            font=("Segoe UI", 25, "bold"),
            fg=WHITE,
            bg=BG
        ).pack(anchor="w")

        tk.Label(
            left,
            text="Monitor human activity in real time",
            font=("Segoe UI", 11),
            fg=MUTED,
            bg=BG
        ).pack(anchor="w", pady=(5, 0))

        right = tk.Frame(header, bg=BG)
        right.pack(side="right")

        self.system_label = tk.Label(
            right,
            text="● SYSTEM ONLINE",
            font=("Segoe UI", 11, "bold"),
            fg=GREEN,
            bg=BG
        )

        self.system_label.pack(
            side="left",
            padx=20
        )

        self.clock_label = tk.Label(
            right,
            font=("Segoe UI", 10, "bold"),
            fg=BLUE,
            bg=BG
        )

        self.clock_label.pack(
            side="right",
            padx=10
        )

        self.update_clock()

    def update_clock(self):

        if self.shutdown_event.is_set():
            return

        self.clock_label.config(
            text=datetime.now().strftime(
                "%d %B %Y\n%I:%M:%S %p"
            )
        )

        self.root.after(
            1000,
            self.update_clock
        )

    # =================================================
    # DASHBOARD ANIMATIONS
    # =================================================

    def animate_dashboard(self):

        if self.shutdown_event.is_set():
            return

        self.animation_phase += 1

        color = self.animation_colors[
            self.animation_phase % len(
                self.animation_colors
            )
        ]

        # Animated logo
        self.logo_icon.config(
            fg=color
        )

        # Pulsing system status
        if self.camera_running:

            if self.stop_event.is_set():

                self.system_label.config(
                    text="● STOPPING",
                    fg=ORANGE
                )

            else:

                self.system_label.config(
                    text="● SYSTEM RUNNING",
                    fg=GREEN if self.animation_phase % 2 else "#86EFAC"
                )

        else:

            self.system_label.config(
                text="● SYSTEM ONLINE",
                fg=GREEN if self.animation_phase % 2 else "#86EFAC"
            )

        # Animate camera live indicator
        if hasattr(self, "live_badge"):

            if self.camera_running and not self.stop_event.is_set():

                self.live_badge.config(
                    text="● LIVE",
                    fg=GREEN if self.animation_phase % 2 else "#86EFAC"
                )

            elif self.stop_event.is_set() and self.camera_running:

                self.live_badge.config(
                    text="● STOPPING",
                    fg=ORANGE
                )

        # Animate people detection card
        if hasattr(self, "people_value"):

            if self.camera_running and not self.stop_event.is_set():

                if self.people_value.cget("text") != "0":

                    self.people_value.config(
                        fg=GREEN if self.animation_phase % 2 else "#86EFAC"
                    )

        # Repeat animation
        self.animation_job = self.root.after(
            550,
            self.animate_dashboard
        )

    # =================================================
    # STATUS CARDS
    # =================================================

    def create_status_section(self):

        frame = tk.Frame(
            self.main_area,
            bg=BG
        )

        frame.pack(
            fill="x",
            padx=20,
            pady=5
        )

        self.people_value = self.create_status_card(
            frame,
            "PEOPLE DETECTED",
            "0",
            "◉",
            BLUE
        )

        self.motion_value = self.create_status_card(
            frame,
            "MOVEMENT",
            "Waiting",
            "⌁",
            PURPLE
        )

        self.alarm_value = self.create_status_card(
            frame,
            "ALARM STATUS",
            "OFF",
            "♢",
            RED
        )

        self.recording_value = self.create_status_card(
            frame,
            "RECORDING",
            "OFF",
            "●",
            GREEN
        )

    def create_status_card(
        self,
        parent,
        title,
        value,
        icon,
        accent
    ):

        card = tk.Frame(
            parent,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5,
            pady=8
        )

        top = tk.Frame(card, bg=PANEL)

        top.pack(
            fill="x",
            padx=14,
            pady=(14, 5)
        )

        tk.Label(
            top,
            text=title,
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=PANEL
        ).pack(side="left")

        tk.Label(
            top,
            text=icon,
            font=("Segoe UI", 18, "bold"),
            fg=accent,
            bg=PANEL
        ).pack(side="right")

        value_label = tk.Label(
            card,
            text=value,
            font=("Segoe UI", 21, "bold"),
            fg=accent,
            bg=PANEL
        )

        value_label.pack(
            anchor="w",
            padx=14,
            pady=(0, 14)
        )

        return value_label

    # =================================================
    # CONTENT
    # =================================================

    def create_content_section(self):

        content = tk.Frame(
            self.main_area,
            bg=BG
        )

        content.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=12
        )

        self.camera_panel = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        self.camera_panel.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.create_camera_panel()

        self.control_panel = tk.Frame(
            content,
            bg=PANEL,
            width=280,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        self.control_panel.pack(
            side="right",
            fill="y",
            padx=(15, 0)
        )

        self.control_panel.pack_propagate(False)

        self.create_control_panel()

    # =================================================
    # CAMERA PANEL
    # =================================================

    def create_camera_panel(self):

        title_frame = tk.Frame(
            self.camera_panel,
            bg=PANEL
        )

        title_frame.pack(
            fill="x",
            padx=20,
            pady=17
        )

        tk.Label(
            title_frame,
            text="◉  LIVE CAMERA MONITOR",
            font=("Segoe UI", 13, "bold"),
            fg=WHITE,
            bg=PANEL
        ).pack(side="left")

        self.live_badge = tk.Label(
            title_frame,
            text="● OFFLINE",
            font=("Segoe UI", 10, "bold"),
            fg=MUTED,
            bg=PANEL_LIGHT,
            padx=12,
            pady=7
        )

        self.live_badge.pack(side="right")

        screen = tk.Frame(
            self.camera_panel,
            bg="#030916",
            highlightbackground="#24466D",
            highlightthickness=1
        )

        screen.pack(
            fill="both",
            expand=True,
            padx=17,
            pady=(0, 16)
        )

        self.camera_display = tk.Label(
            screen,
            text=(
                "\n\n\n◉\n\n"
                "CAMERA NOT STARTED\n\n"
                "Click START CAMERA to begin monitoring"
            ),
            font=("Segoe UI", 15, "bold"),
            fg=MUTED,
            bg="#030916",
            justify="center"
        )

        self.camera_display.pack(
            fill="both",
            expand=True
        )

    # =================================================
    # MONITORING PANEL
    # =================================================

    def create_control_panel(self):

        tk.Label(
            self.control_panel,
            text="●  MONITORING STATUS",
            font=("Segoe UI", 12, "bold"),
            fg=WHITE,
            bg=PANEL
        ).pack(
            anchor="w",
            padx=18,
            pady=(22, 20)
        )

        self.camera_status_value = self.create_info_row(
            "Camera", "Disconnected", BLUE
        )

        self.detection_status_value = self.create_info_row(
            "AI Detection", "Ready", GREEN
        )

        self.alarm_status_value = self.create_info_row(
            "Alarm", "Disabled", RED
        )

        self.recording_status_value = self.create_info_row(
            "Recording", "Stopped", GREEN
        )

        tk.Frame(
            self.control_panel,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=18,
            pady=18
        )

        tk.Label(
            self.control_panel,
            text="▣  RECORDING INFORMATION",
            font=("Segoe UI", 10, "bold"),
            fg=WHITE,
            bg=PANEL
        ).pack(
            anchor="w",
            padx=18,
            pady=(0, 15)
        )

        for title, value, color in [
            ("Video format", "MP4", BLUE),
            ("File name", "Timestamp based", WHITE)
        ]:

            tk.Label(
                self.control_panel,
                text=title,
                font=("Segoe UI", 9),
                fg=MUTED,
                bg=PANEL
            ).pack(anchor="w", padx=18)

            tk.Label(
                self.control_panel,
                text=value,
                font=("Segoe UI", 10, "bold"),
                fg=color,
                bg=PANEL
            ).pack(
                anchor="w",
                padx=18,
                pady=(4, 13)
            )

        tk.Label(
            self.control_panel,
            text=(
                "Recording starts automatically "
                "when the camera starts.\n\n"
                "Recording stops and saves when "
                "you click STOP CAMERA."
            ),
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=PANEL,
            justify="left",
            wraplength=225
        ).pack(
            anchor="w",
            padx=18,
            pady=12
        )

    def create_info_row(self, title, value, color):

        row = tk.Frame(
            self.control_panel,
            bg=PANEL
        )

        row.pack(
            fill="x",
            padx=18,
            pady=9
        )

        tk.Label(
            row,
            text=title,
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=PANEL
        ).pack(side="left")

        value_label = tk.Label(
            row,
            text=value,
            font=("Segoe UI", 10, "bold"),
            fg=color,
            bg=PANEL
        )

        value_label.pack(side="right")

        return value_label

    # =================================================
    # RECORDING HISTORY
    # =================================================

    def create_history_section(self):

        self.history_panel = tk.Frame(
            self.main_area,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        self.history_panel.pack(
            fill="x",
            padx=25,
            pady=(5, 20)
        )

        heading = tk.Frame(
            self.history_panel,
            bg=PANEL
        )

        heading.pack(
            fill="x",
            padx=18,
            pady=15
        )

        tk.Label(
            heading,
            text="▤  RECENT RECORDING HISTORY",
            font=("Segoe UI", 12, "bold"),
            fg=WHITE,
            bg=PANEL
        ).pack(side="left")

        tk.Label(
            heading,
            text="Saved camera sessions",
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=PANEL
        ).pack(side="right")

        columns = (
            "ID",
            "Date and Time",
            "Session",
            "Status",
            "Video File"
        )

        self.history_table = ttk.Treeview(
            self.history_panel,
            columns=columns,
            show="headings",
            height=3
        )

        widths = {
            "ID": 55,
            "Date and Time": 200,
            "Session": 160,
            "Status": 120,
            "Video File": 250
        }

        for column in columns:

            self.history_table.heading(
                column,
                text=column
            )

            self.history_table.column(
                column,
                width=widths[column],
                anchor="center"
            )

        self.history_table.pack(
            fill="x",
            padx=15,
            pady=(0, 15)
        )

    # =================================================
    # NAVIGATION
    # =================================================

    def go_dashboard(self):
        self.main_area.focus_set()

    def go_camera(self):
        self.camera_panel.focus_set()

    def go_history(self):
        self.history_panel.focus_set()

    def open_recordings(self):

        try:
            os.startfile(str(self.recording_folder))

        except Exception as error:

            messagebox.showerror(
                "Recordings",
                str(error)
            )

    def show_settings(self):

        messagebox.showinfo(
            "System Information",
            "AI Human Motion Detection System\n\n"
            "Detection: YOLO11n\n"
            "Camera: OpenCV\n"
            "Alarm: Continuous sound\n"
            "Recording: MP4\n"
            "Language: Python\n"
            "GUI: Tkinter with animations"
        )

    # =================================================
    # ALARM
    # =================================================

    def alarm_loop(self):

        while not self.shutdown_event.is_set():

            if self.alarm_event.wait(timeout=0.2):

                while (
                    self.alarm_event.is_set()
                    and not self.shutdown_event.is_set()
                ):

                    try:

                        winsound.Beep(1500, 500)

                    except RuntimeError as error:

                        print("Alarm Error:", error)
                        self.alarm_event.clear()
                        break

                    self.shutdown_event.wait(0.1)

    # =================================================
    # START CAMERA
    # =================================================

    def start_camera(self):

        if self.shutdown_event.is_set():
            return

        if self.camera_running:
            return

        if (
            self.camera_thread is not None
            and self.camera_thread.is_alive()
        ):
            return

        self.camera_running = True

        self.stop_event.clear()
        self.alarm_event.clear()

        self.start_button.config(
            state="disabled"
        )

        # STOP ALWAYS VISIBLE AND AVAILABLE
        self.stop_button.config(
            state="normal"
        )

        self.live_badge.config(
            text="● STARTING",
            fg=ORANGE
        )

        self.system_label.config(
            text="● SYSTEM STARTING",
            fg=ORANGE
        )

        self.sidebar_status.config(
            text="● Camera Starting",
            fg=ORANGE
        )

        self.camera_status_value.config(
            text="Starting",
            fg=ORANGE
        )

        self.people_value.config(
            text="0",
            fg=BLUE
        )

        self.motion_value.config(
            text="Waiting",
            fg=PURPLE
        )

        self.alarm_value.config(
            text="OFF",
            fg=MUTED
        )

        self.recording_value.config(
            text="STARTING",
            fg=ORANGE
        )

        self.recording_status_value.config(
            text="Starting",
            fg=ORANGE
        )

        self.camera_display.config(
            image="",
            text=(
                "\n\n\n◉\n\n"
                "STARTING CAMERA...\n\n"
                "Loading AI model and connecting webcam"
            ),
            fg=BLUE
        )

        self.camera_display.image = None

        self.camera_thread = threading.Thread(
            target=self.camera_loop,
            daemon=True
        )

        self.camera_thread.start()

    # =================================================
    # START RECORDING
    # =================================================

    def start_recording(self, frame):

        if self.video_writer is not None:
            return

        height, width = frame.shape[:2]

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        self.recording_path = (
            self.recording_folder
            / f"camera_recording_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        fps = 20.0

        if self.camera is not None:

            camera_fps = self.camera.get(
                cv2.CAP_PROP_FPS
            )

            if camera_fps and camera_fps > 1:
                fps = camera_fps

        writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            writer.release()

            print("ERROR: Could not start recording.")

            self.recording_active = False

            self.root.after(
                0,
                lambda: self.recording_status_value.config(
                    text="Failed",
                    fg=RED
                )
            )

            return

        self.video_writer = writer
        self.recording_start_time = datetime.now()
        self.recording_active = True

        print(
            "Recording started:",
            self.recording_path
        )

        self.root.after(
            0,
            self.update_recording_status
        )

    # =================================================
    # STOP RECORDING
    # =================================================

    def stop_recording(self):

        if self.video_writer is None:
            return

        writer = self.video_writer
        self.video_writer = None

        writer.release()

        self.recording_active = False

        saved_path = self.recording_path
        start_time = self.recording_start_time

        self.recording_start_time = None

        print("Recording stopped.")

        if (
            saved_path is not None
            and saved_path.exists()
            and saved_path.stat().st_size > 0
        ):

            print("Video saved:", saved_path)

            if not self.shutdown_event.is_set():

                try:

                    self.root.after(
                        0,
                        lambda p=saved_path, t=start_time:
                        self.add_history(p, t)
                    )

                except RuntimeError:
                    pass

    # =================================================
    # RECORDING STATUS
    # =================================================

    def update_recording_status(self):

        if self.shutdown_event.is_set():
            return

        if self.recording_active:

            self.recording_value.config(
                text="ON",
                fg=GREEN
            )

            self.recording_status_value.config(
                text="Recording",
                fg=GREEN
            )

    # =================================================
    # HISTORY
    # =================================================

    def add_history(self, video_path, start_time):

        if self.shutdown_event.is_set():
            return

        self.history_counter += 1

        date_text = (
            start_time.strftime("%d-%m-%Y %I:%M:%S %p")
            if start_time
            else datetime.now().strftime(
                "%d-%m-%Y %I:%M:%S %p"
            )
        )

        self.history_table.insert(
            "",
            "end",
            values=(
                self.history_counter,
                date_text,
                "Camera Session",
                "Completed",
                video_path.name
            )
        )

    # =================================================
    # CAMERA LOOP
    # =================================================

    def camera_loop(self):

        try:

            print("Loading YOLO model...")

            if self.model is None:
                self.model = YOLO("yolo11n.pt")

            if self.stop_event.is_set():
                return

            print("Opening webcam...")

            self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():

                raise RuntimeError(
                    "Cannot open webcam. Check camera connection."
                )

            print("Webcam connected!")

            self.root.after(
                0,
                self.camera_connected
            )

            while not self.stop_event.is_set():

                success, frame = self.camera.read()

                if not success:

                    raise RuntimeError(
                        "Unable to read webcam frame."
                    )

                # AUTOMATIC RECORDING
                if self.video_writer is None:
                    self.start_recording(frame)

                # HUMAN DETECTION
                results = self.model(
                    frame,
                    conf=0.5,
                    verbose=False
                )

                people_count = 0

                for result in results:

                    for box in result.boxes:

                        class_id = int(box.cls[0])

                        if class_id == 0:

                            people_count += 1

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0]
                            )

                            confidence = float(box.conf[0])

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                frame,
                                f"Person {confidence:.2f}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                # SAVE VIDEO FRAME
                if self.video_writer is not None:
                    self.video_writer.write(frame)

                # CONTINUOUS ALARM
                if people_count > 0:
                    self.alarm_event.set()
                else:
                    self.alarm_event.clear()

                # GUI PREVIEW
                frame_copy = frame.copy()

                try:

                    self.root.after(
                        0,
                        lambda f=frame_copy, c=people_count:
                        self.update_camera_frame(f, c)
                    )

                except RuntimeError:
                    break

        except Exception as error:

            print("Camera Error:", error)

            if not self.shutdown_event.is_set():

                try:

                    self.root.after(
                        0,
                        lambda e=str(error):
                        self.show_camera_error(e)
                    )

                except RuntimeError:
                    pass

        finally:

            self.alarm_event.clear()

            # SAVE RECORDING
            self.stop_recording()

            # RELEASE CAMERA
            if self.camera is not None:

                self.camera.release()
                self.camera = None

            self.camera_running = False

            if not self.shutdown_event.is_set():

                try:

                    self.root.after(
                        0,
                        self.camera_stopped
                    )

                except RuntimeError:
                    pass

    # =================================================
    # CAMERA CONNECTED
    # =================================================

    def camera_connected(self):

        if (
            self.shutdown_event.is_set()
            or not self.camera_running
            or self.stop_event.is_set()
        ):
            return

        self.live_badge.config(
            text="● LIVE",
            fg=GREEN
        )

        self.camera_status_value.config(
            text="Connected",
            fg=GREEN
        )

        self.sidebar_status.config(
            text="● Camera Connected",
            fg=GREEN
        )

    # =================================================
    # CAMERA PREVIEW
    # =================================================

    def update_camera_frame(self, frame, people_count):

        if (
            not self.camera_running
            or self.stop_event.is_set()
            or self.shutdown_event.is_set()
        ):
            return

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)
        image.thumbnail((1100, 600))

        photo = ImageTk.PhotoImage(image=image)

        self.camera_display.config(
            image=photo,
            text=""
        )

        self.camera_display.image = photo

        self.people_value.config(
            text=str(people_count)
        )

        if self.recording_active:

            self.recording_value.config(
                text="ON",
                fg=GREEN
            )

            self.recording_status_value.config(
                text="Recording",
                fg=GREEN
            )

        if people_count > 0:

            self.motion_value.config(
                text="Detected",
                fg=GREEN
            )

            self.alarm_value.config(
                text="ON",
                fg=RED
            )

            self.alarm_status_value.config(
                text="Active",
                fg=RED
            )

        else:

            self.motion_value.config(
                text="No Person",
                fg=BLUE
            )

            self.alarm_value.config(
                text="OFF",
                fg=MUTED
            )

            self.alarm_status_value.config(
                text="Disabled",
                fg=MUTED
            )

    # =================================================
    # CAMERA ERROR
    # =================================================

    def show_camera_error(self, error):

        if self.shutdown_event.is_set():
            return

        self.camera_display.config(
            image="",
            text="CAMERA ERROR\n\n" + error,
            fg=RED
        )

        self.camera_display.image = None

        self.live_badge.config(
            text="● ERROR",
            fg=RED
        )

    # =================================================
    # STOP CAMERA
    # =================================================

    def stop_camera(self):

        if not self.camera_running:
            return

        if self.stop_event.is_set():
            return

        # Stop camera and alarm
        self.stop_event.set()
        self.alarm_event.clear()

        # Keep STOP button visible and accessible
        self.stop_button.config(
            state="normal"
        )

        self.live_badge.config(
            text="● STOPPING",
            fg=ORANGE
        )

        self.system_label.config(
            text="● SYSTEM STOPPING",
            fg=ORANGE
        )

        self.sidebar_status.config(
            text="● Stopping Camera",
            fg=ORANGE
        )

        self.camera_display.config(
            image="",
            text=(
                "\n\n\n◉\n\n"
                "STOPPING CAMERA...\n\n"
                "Saving recording. Please wait."
            ),
            fg=ORANGE
        )

        self.camera_display.image = None

        self.recording_value.config(
            text="SAVING...",
            fg=ORANGE
        )

        self.recording_status_value.config(
            text="Saving",
            fg=ORANGE
        )

        print("Stopping camera and recording...")

    # =================================================
    # CAMERA STOPPED
    # =================================================

    def camera_stopped(self):

        if self.shutdown_event.is_set():
            return

        self.start_button.config(
            state="normal"
        )

        self.stop_button.config(
            state="disabled"
        )

        self.camera_display.config(
            image="",
            text=(
                "\n\n\n◉\n\n"
                "CAMERA STOPPED\n\n"
                "Recording session finished.\n\n"
                "Click START CAMERA to begin again."
            ),
            fg=MUTED
        )

        self.camera_display.image = None

        self.people_value.config(
            text="0",
            fg=BLUE
        )

        self.motion_value.config(
            text="Waiting",
            fg=PURPLE
        )

        self.alarm_value.config(
            text="OFF",
            fg=MUTED
        )

        self.recording_value.config(
            text="OFF",
            fg=MUTED
        )

        self.live_badge.config(
            text="● OFFLINE",
            fg=MUTED
        )

        self.camera_status_value.config(
            text="Disconnected",
            fg=MUTED
        )

        self.alarm_status_value.config(
            text="Disabled",
            fg=MUTED
        )

        self.recording_status_value.config(
            text="Stopped",
            fg=MUTED
        )

        self.sidebar_status.config(
            text="● AI Detection Ready",
            fg=GREEN
        )

        self.system_label.config(
            text="● SYSTEM ONLINE",
            fg=GREEN
        )

        print("Camera stopped and recording saved.")

    # =================================================
    # CLOSE APPLICATION
    # =================================================

    def close_application(self):

        if self.shutdown_event.is_set():
            return

        self.shutdown_event.set()

        self.stop_event.set()
        self.alarm_event.clear()

        if self.animation_job is not None:

            try:
                self.root.after_cancel(
                    self.animation_job
                )
            except tk.TclError:
                pass

        self.check_camera_before_close()

    def check_camera_before_close(self):

        if (
            self.camera_thread is not None
            and self.camera_thread.is_alive()
        ):

            self.root.after(
                100,
                self.check_camera_before_close
            )

        else:

            self.root.destroy()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = HumanMotionDetectionGUI(root)

    root.mainloop()
