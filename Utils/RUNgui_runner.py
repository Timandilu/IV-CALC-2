import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import json
import time
import schedule
import ttkbootstrap as tb
from ttkbootstrap.constants import *

PROMPTS_FILE = "bash_prompts.json"

class BashPromptRunner:
    def __init__(self, root):
        self.root = root
        self.root.title("🖥️ Bash Prompt Runner")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.prompts = self.load_prompts()
        self.selected_prompt = tk.StringVar(value=self.prompts[0] if self.prompts else "")
        self.scheduled_job = None

        # Style
        style = tb.Style("superhero")  # Try "superhero", "flatly", "cyborg", "minty", etc.

        # Title
        title = tb.Label(root, text="🖥️ Bash Prompt Runner", font=("Segoe UI", 22, "bold"), bootstyle=PRIMARY)
        title.pack(pady=(18, 0))

        # Prompt add/remove
        frame = tb.Frame(root, bootstyle=SECONDARY)
        frame.pack(pady=15, padx=18, fill="x")
        tb.Label(frame, text="Add New Prompt:", font=("Segoe UI", 11, "bold"), bootstyle=INFO).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.prompt_entry = tb.Entry(frame, width=55, font=("Segoe UI", 10))
        self.prompt_entry.grid(row=1, column=0, sticky="ew", pady=3, padx=(0, 6))
        add_btn = tb.Button(frame, text="➕ Add", bootstyle=SUCCESS, width=10, command=self.add_prompt)
        add_btn.grid(row=1, column=1, padx=2)
        remove_btn = tb.Button(frame, text="🗑 Remove", bootstyle=DANGER, width=12, command=self.remove_prompt)
        remove_btn.grid(row=1, column=2, padx=2)

        # Prompts list with run buttons
        list_frame = tb.Labelframe(root, text="Saved Prompts", bootstyle=INFO)
        list_frame.pack(padx=18, pady=8, fill="x", expand=False)
        self.prompts_canvas = tk.Canvas(list_frame, height=140, bg=style.colors.bg, highlightthickness=0)
        self.prompts_canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tb.Scrollbar(list_frame, orient="vertical", command=self.prompts_canvas.yview, bootstyle=ROUND)
        scrollbar.pack(side="right", fill="y")
        self.prompts_canvas.configure(yscrollcommand=scrollbar.set)
        self.prompts_inner = tb.Frame(self.prompts_canvas)
        self.prompts_canvas.create_window((0, 0), window=self.prompts_inner, anchor="nw")
        self.prompts_inner.bind("<Configure>", lambda e: self.prompts_canvas.configure(scrollregion=self.prompts_canvas.bbox("all")))
        self.prompt_buttons = []
        self.render_prompt_buttons()

        # Schedule/cancel buttons
        btn_frame = tb.Frame(root)
        btn_frame.pack(pady=10)
        sched_btn = tb.Button(btn_frame, text="⏰ Schedule Every Minute", bootstyle=WARNING, width=22, command=self.schedule_prompt)
        sched_btn.grid(row=0, column=0, padx=8)
        cancel_btn = tb.Button(btn_frame, text="✖ Cancel Schedule", bootstyle=SECONDARY, width=18, command=self.cancel_schedule)
        cancel_btn.grid(row=0, column=1, padx=8)

        # Log area
        log_label = tb.Label(root, text="Logs (APPLOG: only):", font=("Segoe UI", 11, "bold"), bootstyle=PRIMARY)
        log_label.pack(anchor="w", padx=22, pady=(8, 0))
        self.log_area = scrolledtext.ScrolledText(root, width=90, height=16, state='disabled',
                                                  bg="#181c20", fg="#aefbaf", font=("Consolas", 11), bd=0, relief="flat")
        self.log_area.pack(padx=18, pady=(0, 10), fill="both", expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tb.Label(root, textvariable=self.status_var, anchor="w", bootstyle=INVERSE)
        status_bar.pack(side="bottom", fill="x")

        # Scheduler thread
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()

    def render_prompt_buttons(self):
        # Clear previous
        for widget in self.prompts_inner.winfo_children():
            widget.destroy()
        self.prompt_buttons.clear()
        for idx, prompt in enumerate(self.prompts):
            prompt_label = tb.Radiobutton(self.prompts_inner, text=prompt, variable=self.selected_prompt, value=prompt,
                                          bootstyle=INFO, width=60)
            prompt_label.grid(row=idx, column=0, sticky="w", padx=2, pady=2)
            run_btn = tb.Button(self.prompts_inner, text="▶ Run", bootstyle=SUCCESS, width=8,
                                command=lambda p=prompt: self.run_prompt_threaded(p))
            run_btn.grid(row=idx, column=1, padx=2)
            self.prompt_buttons.append((prompt_label, run_btn))

    def load_prompts(self):
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r") as f:
                return json.load(f)
        return ["echo APPLOG:Hello World"]

    def save_prompts(self):
        with open(PROMPTS_FILE, "w") as f:
            json.dump(self.prompts, f)

    def add_prompt(self):
        prompt = self.prompt_entry.get().strip()
        if prompt and prompt not in self.prompts:
            self.prompts.append(prompt)
            self.save_prompts()
            self.selected_prompt.set(prompt)
            self.render_prompt_buttons()
            self.append_log(f"Added prompt: {prompt}\n")
            self.status_var.set("Prompt added.")
        else:
            self.status_var.set("Prompt is empty or already exists.")

    def remove_prompt(self):
        prompt = self.selected_prompt.get()
        if prompt in self.prompts:
            self.prompts.remove(prompt)
            self.save_prompts()
            if self.prompts:
                self.selected_prompt.set(self.prompts[0])
            else:
                self.selected_prompt.set("")
            self.render_prompt_buttons()
            self.append_log(f"Removed prompt: {prompt}\n")
            self.status_var.set("Prompt removed.")
        else:
            self.status_var.set("No prompt selected.")

    def run_prompt_threaded(self, prompt):
        self.append_log(f"Running: {prompt}\n")
        self.status_var.set("Running prompt...")
        threading.Thread(target=self.run_prompt, args=(prompt,), daemon=True).start()

    def run_prompt(self, prompt):
        try:
            proc = subprocess.Popen(
                prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True
            )
            for line in proc.stdout:
                if line.startswith("APPLOG:"):
                    self.append_log(line.replace("APPLOG:", "", 1))
            proc.wait()
            self.append_log(f"\nPrompt finished with exit code {proc.returncode}\n")
            self.status_var.set(f"Prompt finished (exit code {proc.returncode})")
        except Exception as e:
            self.append_log(f"Error running prompt: {e}\n")
            self.status_var.set("Error running prompt.")

    def append_log(self, text):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, text)
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def schedule_prompt(self):
        prompt = self.selected_prompt.get()
        if not prompt:
            messagebox.showwarning("No prompt selected", "Please select a prompt to schedule.")
            self.status_var.set("No prompt selected for scheduling.")
            return
        if self.scheduled_job:
            messagebox.showinfo("Already scheduled", "A prompt is already scheduled.")
            self.status_var.set("A prompt is already scheduled.")
            return
        self.scheduled_job = schedule.every(1).minutes.do(lambda: self.run_prompt(prompt))
        self.append_log(f"Scheduled prompt every minute: {prompt}\n")
        self.status_var.set("Prompt scheduled every minute.")

    def cancel_schedule(self):
        if self.scheduled_job:
            schedule.cancel_job(self.scheduled_job)
            self.scheduled_job = None
            self.append_log("Cancelled scheduled prompt.\n")
            self.status_var.set("Scheduled prompt cancelled.")
        else:
            self.append_log("No scheduled prompt to cancel.\n")
            self.status_var.set("No scheduled prompt to cancel.")

    def run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    root = tb.Window(themename="superhero")
    app = BashPromptRunner(root)
    root.mainloop()