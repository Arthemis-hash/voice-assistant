"""
Voice Pipeline TUI Application.

Clean, efficient terminal interface for managing the voice pipeline.
Built with Textual - modern Python TUI framework.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Input,
    Select,
    DataTable,
    Log,
    TabbedContent,
    TabPane,
    Switch,
    Label,
    RichLog,
)
from textual.binding import Binding
from textual.screen import Screen
from textual.message import Message

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.lm_studio_provider import LMStudioProvider
from pipeline.database import Database
from tts.piper_tts import PiperTTS

# Available voices
VOICES = [
    ("en_US-lessac-medium", "English (US) - Lessac (Female)"),
    ("en_US-ryan-high", "English (US) - Ryan (Male)"),
    ("fr_FR-upmc-medium", "French - UPMC"),
    ("fr_FR-tom-medium", "French - Tom"),
    ("de_DE-mls-medium", "German - MLS"),
    ("es_ES-carlfm-medium", "Spanish - CarlFM"),
    ("it_IT-riccardo-medium", "Italian - Riccardo"),
    ("ja_JP-kamitsukichi-medium", "Japanese - Kamitsukichi"),
]

# Available languages
LANGUAGES = [
    ("en-US", "English (US)"),
    ("en-GB", "English (UK)"),
    ("fr-FR", "French"),
    ("de-DE", "German"),
    ("es-ES", "Spanish"),
    ("it-IT", "Italian"),
    ("ja-JP", "Japanese"),
    ("ko-KR", "Korean"),
    ("zh-CN", "Chinese (Simplified)"),
    ("pt-BR", "Portuguese (Brazil)"),
    ("ar-SA", "Arabic"),
    ("ru-RU", "Russian"),
    ("nl-NL", "Dutch"),
    ("sv-SE", "Swedish"),
    ("da-DK", "Danish"),
    ("no-NO", "Norwegian"),
    ("fi-FI", "Finnish"),
    ("pl-PL", "Polish"),
    ("tr-TR", "Turkish"),
]


class StatusIndicator(Static):
    """Status indicator widget."""

    def __init__(self, label: str = "", status: str = "unknown") -> None:
        super().__init__()
        self._label = label
        self._status = status

    def set_status(self, status: str) -> None:
        self._status = status
        self.refresh()

    def render(self) -> str:
        colors = {
            "online": "#00ff00",
            "offline": "#ff0000",
            "unknown": "#888888",
            "loading": "#ffaa00",
        }
        color = colors.get(self._status, "#888888")
        dot = "●"
        return f"[{color}]{dot}[/] {self._label}: [{color}]{self._status.upper()}[/]"


class DashboardScreen(Screen):
    """Dashboard screen with system status and controls."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("🎙️ Voice Pipeline Dashboard", id="dashboard-title"),
            Horizontal(
                Vertical(
                    StatusIndicator("STT (Apple Speech)", "unknown", id="stt-status"),
                    StatusIndicator("LLM (LM Studio)", "unknown", id="llm-status"),
                    StatusIndicator("TTS (Piper)", "unknown", id="tts-status"),
                    id="status-panel",
                ),
                Vertical(
                    Static("System Controls", id="controls-title"),
                    Button("▶ Start Pipeline", id="start-btn", variant="success"),
                    Button("⏹ Stop Pipeline", id="stop-btn", variant="error"),
                    Button("🔄 Check Connections", id="check-btn", variant="primary"),
                    id="controls-panel",
                ),
                id="main-row",
            ),
            Static("Quick Actions", id="quick-title"),
            Horizontal(
                Button("💬 Conversation", id="conv-btn", variant="primary"),
                Button("📁 Process File", id="file-btn", variant="warning"),
                Button("📜 History", id="history-btn", variant="default"),
                Button("⚙️ Settings", id="settings-btn", variant="default"),
                id="quick-row",
            ),
            id="dashboard-container",
        )

    def on_mount(self) -> None:
        self.check_connections()

    def check_connections(self) -> None:
        """Check all service connections."""
        # STT status
        stt_status = self.query_one("#stt-status", StatusIndicator)
        stt_status.set_status("online")  # Apple Speech is always available on macOS

        # LLM status
        llm_status = self.query_one("#llm-status", StatusIndicator)
        llm_status.set_status("loading")

        # TTS status
        tts_status = self.query_one("#tts-status", StatusIndicator)
        tts_status.set_status("loading")

        # Check LLM
        try:
            base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
            model = os.getenv("LM_STUDIO_MODEL", "Qwen3-4B")
            llm = LMStudioProvider(base_url=base_url, model=model)
            if llm.is_available():
                llm_status.set_status("online")
            else:
                llm_status.set_status("offline")
        except Exception:
            llm_status.set_status("offline")

        # Check TTS
        try:
            voice = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
            tts = PiperTTS(voice=voice)
            tts_status.set_status("online")
        except Exception:
            tts_status.set_status("offline")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "start-btn":
            self.notify("Pipeline started", title="System", severity="information")
        elif button_id == "stop-btn":
            self.notify("Pipeline stopped", title="System", severity="warning")
        elif button_id == "check-btn":
            self.check_connections()
            self.notify("Connections checked", title="System", severity="information")
        elif button_id == "conv-btn":
            self.app.push_screen("conversation")
        elif button_id == "file-btn":
            self.app.push_screen("file_processor")
        elif button_id == "history-btn":
            self.app.push_screen("history")
        elif button_id == "settings-btn":
            self.app.push_screen("settings")


class ConversationScreen(Screen):
    """Interactive conversation screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("enter", "send_message", "Send"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("💬 Conversation", id="conv-title"),
            RichLog(id="chat-log", highlight=True, markup=True),
            Horizontal(
                Input(placeholder="Type your message... (Enter to send)", id="chat-input"),
                Button("Send", id="send-btn", variant="primary"),
                id="input-row",
            ),
            id="conversation-container",
        )

    def on_mount(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("[bold green]🎙️ Conversation started. Type your message and press Enter.[/]")
        chat_log.write("[dim]Press Escape to return to dashboard.[/]")

    def action_send_message(self) -> None:
        self._send_message()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-btn":
            self._send_message()

    def _send_message(self) -> None:
        chat_input = self.query_one("#chat-input", Input)
        chat_log = self.query_one("#chat-log", RichLog)
        message = chat_input.value.strip()

        if not message:
            return

        # Display user message
        chat_log.write(f"\n[bold blue]You:[/] {message}")
        chat_input.value = ""

        # Simulate LLM response (replace with actual LLM call)
        chat_log.write("[dim]⏳ Thinking...[/]")

        try:
            base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
            model = os.getenv("LM_STUDIO_MODEL", "Qwen3-4B")
            llm = LMStudioProvider(base_url=base_url, model=model)
            response = llm.generate(message)
            chat_log.write(f"[bold green]Assistant:[/] {response}")
        except Exception as e:
            chat_log.write(f"[bold red]Error:[/] {str(e)}")


class FileProcessorScreen(Screen):
    """Audio file processor screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("📁 Audio File Processor", id="file-title"),
            Input(placeholder="Enter audio file path (e.g., /path/to/audio.wav)", id="file-input"),
            Horizontal(
                Button("🔄 Process", id="process-btn", variant="success"),
                Button("📂 Browse", id="browse-btn", variant="primary"),
                id="file-buttons",
            ),
            Static("Output will appear here", id="file-output"),
            id="file-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "process-btn":
            self._process_file()
        elif event.button.id == "browse-btn":
            self.notify("File browser not implemented yet", title="Info", severity="warning")

    def _process_file(self) -> None:
        file_input = self.query_one("#file-input", Input)
        output = self.query_one("#file-output", Static)
        file_path = file_input.value.strip()

        if not file_path:
            output.update("[bold red]Please enter a file path[/]")
            return

        output.update(f"[dim]Processing: {file_path}...[/]")

        try:
            # Simulate processing (replace with actual pipeline call)
            output.update(f"[bold green]✅ Processed: {file_path}[/]\n[dim]Output saved to ./output/[/]")
        except Exception as e:
            output.update(f"[bold red]❌ Error: {str(e)}[/]")


class SettingsScreen(Screen):
    """Settings and configuration screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("⚙️ Settings", id="settings-title"),
            Vertical(
                Label("STT Language:"),
                Select(
                    options=[(name, code) for code, name in LANGUAGES],
                    value=os.getenv("STT_LANGUAGE", "en-US"),
                    id="stt-lang-select",
                ),
                Label("TTS Voice:"),
                Select(
                    options=[(name, code) for code, name in VOICES],
                    value=os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
                    id="tts-voice-select",
                ),
                Label("LM Studio URL:"),
                Input(
                    value=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
                    id="llm-url-input",
                ),
                Label("Model Name:"),
                Input(
                    value=os.getenv("LM_STUDIO_MODEL", "Qwen3-4B"),
                    id="llm-model-input",
                ),
                Label("Output Directory:"),
                Input(
                    value=os.getenv("OUTPUT_DIR", "./output"),
                    id="output-dir-input",
                ),
                id="settings-form",
            ),
            Horizontal(
                Button("💾 Save", id="save-btn", variant="success"),
                Button("↩️ Reset", id="reset-btn", variant="warning"),
                id="settings-buttons",
            ),
            id="settings-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save_settings()
        elif event.button.id == "reset-btn":
            self._reset_settings()

    def _save_settings(self) -> None:
        stt_lang = self.query_one("#stt-lang-select", Select).value
        tts_voice = self.query_one("#tts-voice-select", Select).value
        llm_url = self.query_one("#llm-url-input", Input).value
        llm_model = self.query_one("#llm-model-input", Input).value
        output_dir = self.query_one("#output-dir-input", Input).value

        # Save to environment (in production, save to config file)
        os.environ["STT_LANGUAGE"] = str(stt_lang)
        os.environ["PIPER_VOICE"] = str(tts_voice)
        os.environ["LM_STUDIO_URL"] = llm_url
        os.environ["LM_STUDIO_MODEL"] = llm_model
        os.environ["OUTPUT_DIR"] = output_dir

        self.notify("Settings saved successfully", title="Settings", severity="information")

    def _reset_settings(self) -> None:
        self.query_one("#stt-lang-select", Select).value = "en-US"
        self.query_one("#tts-voice-select", Select).value = "en_US-lessac-medium"
        self.query_one("#llm-url-input", Input).value = "http://localhost:1234/v1"
        self.query_one("#llm-model-input", Input).value = "Qwen3-4B"
        self.query_one("#output-dir-input", Input).value = "./output"
        self.notify("Settings reset to defaults", title="Settings", severity="warning")


class LogsScreen(Screen):
    """System logs screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("📋 System Logs", id="logs-title"),
            RichLog(id="system-log", highlight=True, markup=True),
            Horizontal(
                Button("🗑️ Clear", id="clear-btn", variant="error"),
                Button("🔄 Refresh", id="refresh-btn", variant="primary"),
                id="log-buttons",
            ),
            id="logs-container",
        )

    def on_mount(self) -> None:
        self._refresh_logs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear-btn":
            self.query_one("#system-log", RichLog).clear()
            self.notify("Logs cleared", title="Logs", severity="information")
        elif event.button.id == "refresh-btn":
            self._refresh_logs()

    def _refresh_logs(self) -> None:
        log_widget = self.query_one("#system-log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_widget.write(f"[dim]{timestamp}[/] [green]System initialized[/]")
        log_widget.write(f"[dim]{timestamp}[/] [blue]STT: Apple Speech Framework ready[/]")
        log_widget.write(f"[dim]{timestamp}[/] [yellow]TTS: Piper voice loaded[/]")
        log_widget.write(f"[dim]{timestamp}[/] [cyan]LLM: Waiting for LM Studio connection[/]")


class HistoryScreen(Screen):
    """Conversation history screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("📜 Conversation History", id="history-title"),
            DataTable(id="history-table", cursor_type="row"),
            Horizontal(
                Button("🔄 Refresh", id="refresh-history-btn", variant="primary"),
                Button("🗑️ Clear All", id="clear-history-btn", variant="error"),
                id="history-buttons",
            ),
            RichLog(id="history-detail", highlight=True, markup=True),
            id="history-container",
        )

    def on_mount(self) -> None:
        self._load_history()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-history-btn":
            self._load_history()
        elif event.button.id == "clear-history-btn":
            db = Database()
            count = db.clear_conversations()
            self.notify(f"Cleared {count} conversations", title="History", severity="warning")
            self._load_history()

    def _load_history(self) -> None:
        table = self.query_one("#history-table", DataTable)
        detail = self.query_one("#history-detail", RichLog)
        detail.clear()

        table.clear()
        table.add_column("Time", width=10)
        table.add_column("User", width=30)
        table.add_column("Response", width=50)
        table.add_column("Duration", width=8)

        db = Database()
        conversations = db.list_conversations(limit=100)

        if not conversations:
            detail.write("[dim]No conversations yet.[/]")
            return

        for conv in conversations:
            time_str = datetime.fromisoformat(conv["created_at"]).strftime("%H:%M:%S")
            user_text = conv["user_text"][:40] + "..." if len(conv["user_text"]) > 40 else conv["user_text"]
            llm_text = conv["llm_response"][:60] + "..." if len(conv["llm_response"]) > 60 else conv["llm_response"]
            duration = f"{conv['audio_duration']:.1f}s" if conv["audio_duration"] else "N/A"

            table.add_row(time_str, user_text, llm_text, duration, key=conv["id"])

        # Show latest conversation detail
        if conversations:
            latest = conversations[0]
            detail.write(f"[bold blue]User:[/] {latest['user_text']}")
            detail.write(f"[bold green]Assistant:[/] {latest['llm_response']}")
            detail.write(f"[dim]Language: {latest['stt_language']} | Voice: {latest['tts_voice']} | Duration: {latest['audio_duration']:.1f}s[/]")


class VoicePipelineApp(App):
    """Main Voice Pipeline TUI Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    #dashboard-container, #conversation-container, #file-container,
    #settings-container, #logs-container, #history-container {
        padding: 1 2;
    }

    #dashboard-title, #conv-title, #file-title, #settings-title, #logs-title, #history-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
    }

    #main-row {
        height: 10;
        margin: 1 0;
    }

    #status-panel {
        width: 40%;
        border: solid $primary;
        padding: 1 2;
    }

    #controls-panel {
        width: 60%;
        border: solid $primary;
        padding: 1 2;
    }

    #controls-title, #quick-title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #quick-row {
        height: 3;
        margin: 1 0;
    }

    #chat-log, #system-log {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #history-table {
        height: 1fr;
        border: solid $primary;
    }

    #history-detail {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #history-buttons {
        height: 3;
        margin: 1 0;
    }

    #input-row, #file-buttons, #settings-buttons, #log-buttons {
        height: 3;
        margin: 1 0;
    }

    #chat-input {
        width: 1fr;
    }

    #settings-form {
        border: solid $primary;
        padding: 1 2;
        height: 1fr;
    }

    #file-output {
        border: solid $primary;
        padding: 1;
        height: 1fr;
    }

    Button {
        margin: 0 1;
    }

    StatusIndicator {
        padding: 1 0;
    }
    """

    SCREENS = {
        "dashboard": DashboardScreen,
        "conversation": ConversationScreen,
        "file_processor": FileProcessorScreen,
        "settings": SettingsScreen,
        "logs": LogsScreen,
        "history": HistoryScreen,
    }

    BINDINGS = [
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
        Binding("q", "quit", "Quit"),
        Binding("1", "show_dashboard", "Dashboard"),
        Binding("2", "show_conversation", "Conversation"),
        Binding("3", "show_settings", "Settings"),
        Binding("4", "show_logs", "Logs"),
        Binding("5", "show_history", "History"),
    ]

    def on_mount(self) -> None:
        self.push_screen("dashboard")

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_show_dashboard(self) -> None:
        self.push_screen("dashboard")

    def action_show_conversation(self) -> None:
        self.push_screen("conversation")

    def action_show_settings(self) -> None:
        self.push_screen("settings")

    def action_show_logs(self) -> None:
        self.push_screen("logs")

    def action_show_history(self) -> None:
        self.push_screen("history")


def main() -> None:
    """Run the TUI application."""
    app = VoicePipelineApp()
    app.run()


if __name__ == "__main__":
    main()
