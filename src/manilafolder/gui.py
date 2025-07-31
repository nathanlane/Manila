# ───────────────────────── src/manilafolder/gui.py ─────────────────────────
"""
PySimpleGUI interface with retro-monochrome theme.
"""

import threading
from pathlib import Path
from typing import Any, List, Optional

import PySimpleGUI as sg

from .config import Config
from .db import (
    create_vector_store,
    get_collection_stats,
    is_valid_chroma_db,
    open_vector_store,
)
from .ingest import ingest_pdfs
from .logging_utils import log_error


def setup_retro_theme() -> None:
    """Set up the custom RetroMono theme."""

    # Define RetroMono theme colors
    retro_theme = {
        "BACKGROUND": "#E5E5E5",
        "TEXT": "black",
        "INPUT": "#FFFFFF",
        "TEXT_INPUT": "black",
        "SCROLL": "#D0D0D0",
        "BUTTON": ("black", "#D0D0D0"),
        "PROGRESS": ("black", "#A0A0A0"),
        "BORDER": 1,
        "SLIDER_DEPTH": 0,
        "PROGRESS_DEPTH": 0,
    }

    # Add the theme to PySimpleGUI
    sg.theme_add_new("RetroMono", retro_theme)
    sg.theme("RetroMono")

    # Set global options
    sg.set_options(
        font=("JetBrains Mono", 11),
        element_padding=(6, 4),
        button_element_size=(12, 1),
        border_width=1,
        margins=(10, 10),
    )


class ManilaFolderGUI:
    """Main GUI class for ManilaFolder application."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the GUI.

        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or Config()
        self.collection = None
        self.db_path = None
        self.window = None
        self.file_data = []

        setup_retro_theme()

    def create_startup_layout(self) -> List[List[Any]]:
        """Create the startup screen layout.

        Returns:
            PySimpleGUI layout for startup screen
        """
        return [
            [
                sg.Text(
                    "ManilaFolder",
                    font=("JetBrains Mono", 14, "bold"),
                    justification="center",
                    expand_x=True,
                )
            ],
            [
                sg.Text(
                    "Digital Document Organization with Vector Search",
                    justification="center",
                    expand_x=True,
                )
            ],
            [sg.VPush()],
            [sg.Button("Create New DB", key="-CREATE-", size=(20, 2))],
            [sg.VPush()],
            [sg.Button("Open Existing DB", key="-OPEN-", size=(20, 2))],
            [sg.VPush()],
        ]

    def create_main_layout(self) -> List[List[Any]]:
        """Create the main workspace layout.

        Returns:
            PySimpleGUI layout for main workspace
        """
        # Left pane - file management
        left_pane = [
            [
                sg.Frame(
                    "Manila Folders",
                    [
                        [
                            sg.Table(
                                values=[],
                                headings=["Filename", "Pages", "Chunks"],
                                col_widths=[25, 8, 8],
                                num_rows=15,
                                key="-FILE_TABLE-",
                                enable_events=True,
                                expand_x=True,
                                expand_y=True,
                            )
                        ],
                        [
                            sg.Button("Add PDFs...", key="-ADD_FILES-"),
                            sg.Button("Refresh", key="-REFRESH-"),
                        ],
                    ],
                    expand_x=True,
                    expand_y=True,
                )
            ]
        ]

        # Right pane - log and progress
        right_pane = [
            [
                sg.Frame(
                    "Activity Log",
                    [
                        [
                            sg.Multiline(
                                size=(40, 10),
                                key="-LOG-",
                                disabled=True,
                                autoscroll=True,
                                expand_x=True,
                                expand_y=True,
                            )
                        ],
                        [
                            sg.ProgressBar(
                                max_value=100,
                                orientation="h",
                                size=(30, 20),
                                key="-PROGRESS-",
                                expand_x=True,
                            )
                        ],
                        [sg.Text("Ready", key="-STATUS-", size=(30, 1))],
                    ],
                    expand_x=True,
                    expand_y=True,
                )
            ]
        ]

        return [
            [
                sg.Text(
                    f"Database: {self.db_path}",
                    key="-DB_PATH-",
                    font=("JetBrains Mono", 9),
                )
            ],
            [
                sg.Column(left_pane, expand_x=True, expand_y=True),
                sg.Column(right_pane, expand_x=True, expand_y=True),
            ],
            [
                sg.Button("Close Database", key="-CLOSE_DB-"),
                sg.Push(),
                sg.Button("Exit", key="-EXIT-"),
            ],
        ]

    def log_message(self, message: str) -> None:
        """Add a message to the GUI log.

        Args:
            message: Message to display
        """
        if self.window:
            try:
                current_text = self.window["-LOG-"].get()
                new_text = current_text + f"{message}\n"
                self.window["-LOG-"].update(new_text)
                # Auto-scroll to bottom
                self.window["-LOG-"].set_cursor(len(new_text))
            except Exception:
                # Fallback: just update with the message
                self.window["-LOG-"].update(f"{message}\n")

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress bar and status.

        Args:
            current: Current progress value
            total: Total progress value
            message: Status message
        """
        if self.window:
            if total > 0:
                progress = int((current / total) * 100)
                self.window["-PROGRESS-"].update(progress)

            if message:
                self.window["-STATUS-"].update(message)

            self.window.refresh()

    def refresh_file_table(self) -> None:
        """Refresh the file table with current database contents."""
        if not self.collection:
            return

        try:
            # Get collection statistics
            stats = get_collection_stats(self.collection)

            # Get sample documents to build file list
            try:
                results = self.collection.get(limit=1000)
                metadatas = results.get("metadatas", [])

                # Group by source file
                file_stats = {}
                for metadata in metadatas:
                    if metadata and "source" in metadata:
                        source = metadata["source"]
                        filename = Path(source).name

                        if filename not in file_stats:
                            file_stats[filename] = {"pages": set(), "chunks": 0}

                        if "page" in metadata:
                            file_stats[filename]["pages"].add(metadata["page"])

                        file_stats[filename]["chunks"] += 1

                # Convert to table data
                table_data = []
                for filename, stats in file_stats.items():
                    table_data.append([filename, len(stats["pages"]), stats["chunks"]])

                self.file_data = table_data

                if self.window:
                    self.window["-FILE_TABLE-"].update(values=table_data)

                self.log_message(
                    f"Database contains {len(file_stats)} files, "
                    f"{sum(len(s['pages']) for s in file_stats.values())} pages, "
                    f"{sum(s['chunks'] for s in file_stats.values())} chunks"
                )

            except Exception as e:
                self.log_message(f"Error refreshing file list: {str(e)}")

        except Exception as e:
            log_error("Failed to refresh file table", e, self.config)
            self.log_message(f"Error: {str(e)}")

    def handle_create_db(self) -> bool:
        """Handle creating a new database.

        Returns:
            True if database created successfully, False otherwise
        """
        folder_name = sg.popup_get_text(
            "Enter folder name for new database:",
            "Create New Database",
            default_text="MyManilaFolder",
        )

        if not folder_name:
            return False

        # Let user choose parent directory
        parent_dir = sg.popup_get_folder("Choose location for new database:")
        if not parent_dir:
            return False

        db_path = Path(parent_dir) / folder_name

        try:
            self.collection = create_vector_store(str(db_path), self.config)
            self.db_path = str(db_path)
            self.log_message(f"Created new database at {db_path}")
            return True

        except FileExistsError:
            sg.popup_error(f"Database already exists at {db_path}")
            return False
        except Exception as e:
            sg.popup_error(f"Failed to create database: {str(e)}")
            return False

    def handle_open_db(self) -> bool:
        """Handle opening an existing database.

        Returns:
            True if database opened successfully, False otherwise
        """
        # Custom folder browser that filters for ChromaDB directories
        selected_path = sg.popup_get_folder(
            "Select folder containing ChromaDB:", initial_folder=str(Path.home())
        )

        if not selected_path:
            return False

        db_path = Path(selected_path)

        # Validate it's a ChromaDB
        if not is_valid_chroma_db(db_path):
            sg.popup_error(
                f"No valid ChromaDB found at {db_path}\n\n"
                "Please select a folder containing a ChromaDB database."
            )
            return False

        try:
            self.collection = open_vector_store(str(db_path), self.config)
            self.db_path = str(db_path)
            self.log_message(f"Opened database at {db_path}")
            return True

        except Exception as e:
            sg.popup_error(f"Failed to open database: {str(e)}")
            return False

    def handle_add_files(self) -> None:
        """Handle adding PDF files to the database."""
        if not self.collection:
            return

        # File dialog for PDFs
        file_paths = sg.popup_get_file(
            "Select PDF files to add:",
            file_types=[("PDF Files", "*.pdf")],
            multiple_files=True,
        )

        if not file_paths:
            return

        # Convert single file path to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        self.log_message(f"Starting ingestion of {len(file_paths)} files...")

        # Run ingestion in separate thread to keep GUI responsive
        def ingestion_worker():
            try:
                stats = ingest_pdfs(
                    file_paths,
                    self.collection,
                    self.config,
                    progress_callback=self.update_progress,
                )

                # Update GUI on completion
                def update_gui():
                    self.log_message(
                        f"Ingestion complete: {stats['files_processed']} files "
                        f"processed, {stats['total_chunks']} chunks added"
                    )

                    if stats["failed_files"]:
                        self.log_message(
                            f"Failed files: "
                            f"{', '.join(Path(f).name for f in stats['failed_files'])}"
                        )

                    self.refresh_file_table()
                    self.update_progress(0, 1, "Ready")

                # Schedule GUI update on main thread
                self.window.write_event_value("-INGESTION_COMPLETE-", update_gui)

            except Exception as e:
                error_msg = f"Ingestion failed: {str(e)}"
                log_error("PDF ingestion failed", e, self.config)

                def show_error():
                    self.log_message(error_msg)
                    self.update_progress(0, 1, "Error")

                self.window.write_event_value("-INGESTION_ERROR-", show_error)

        # Start ingestion thread
        thread = threading.Thread(target=ingestion_worker, daemon=True)
        thread.start()

    def run_startup(self) -> bool:
        """Run the startup screen.

        Returns:
            True if should continue to main app, False if should exit
        """
        layout = self.create_startup_layout()

        self.window = sg.Window(
            "ManilaFolder - Digital Filing Cabinet",
            layout,
            size=(400, 300),
            resizable=False,
            finalize=True,
        )

        # Set window icon (placeholder comment for 1-bit bitmap)
        # self.window.TKroot.iconbitmap('manila_folder_icon.ico')

        while True:
            event, values = self.window.read()

            if event in (sg.WIN_CLOSED, "-EXIT-"):
                self.window.close()
                return False

            elif event == "-CREATE-":
                if self.handle_create_db():
                    self.window.close()
                    return True

            elif event == "-OPEN-":
                if self.handle_open_db():
                    self.window.close()
                    return True

    def run_main(self) -> None:
        """Run the main application window."""
        layout = self.create_main_layout()

        self.window = sg.Window(
            "ManilaFolder - Digital Filing Cabinet",
            layout,
            size=(800, 600),
            resizable=False,
            finalize=True,
        )

        # Set window icon (placeholder comment for 1-bit bitmap)
        # self.window.TKroot.iconbitmap('manila_folder_icon.ico')

        # Initial refresh
        self.refresh_file_table()

        while True:
            event, values = self.window.read(timeout=100)

            if event in (sg.WIN_CLOSED, "-EXIT-"):
                break

            elif event == "-CLOSE_DB-":
                self.collection = None
                self.db_path = None
                self.window.close()
                if self.run_startup():
                    self.run_main()
                break

            elif event == "-ADD_FILES-":
                self.handle_add_files()

            elif event == "-REFRESH-":
                self.refresh_file_table()

            elif event == "-INGESTION_COMPLETE-":
                if values and callable(values[event]):
                    values[event]()

            elif event == "-INGESTION_ERROR-":
                if values and callable(values[event]):
                    values[event]()

        self.window.close()

    def run(self) -> None:
        """Run the complete application."""
        if self.run_startup():
            self.run_main()


def create_gui(config: Optional[Config] = None) -> ManilaFolderGUI:
    """Create and return a ManilaFolder GUI instance.

    Args:
        config: Configuration object, uses defaults if None

    Returns:
        ManilaFolderGUI instance
    """
    return ManilaFolderGUI(config)
