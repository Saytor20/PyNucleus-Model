"""
PyNucleus Interactive Menu System
Enhanced UX with clear post-command navigation options
"""

import sys
from typing import Callable, Optional
from rich.console import Console

# Initialize console for rich formatting
console = Console()

# Import the enhanced error handler
from pynucleus.utils.error_handler import error_handler

@error_handler
def post_command_options(context_menu_func: Callable[[], None]):
    """
    Provide intuitive navigation after command execution.
    
    Args:
        context_menu_func: Function to call when user chooses to repeat context menu
    """
    console.print("\n[bold blue]" + "─" * 60 + "[/bold blue]")
    console.print("[bold cyan]🎯 Post-Command Navigation[/bold cyan]")
    console.print("[bold blue]" + "─" * 60 + "[/bold blue]")
    console.print("[bold green]m[/bold green]: Return to Main Menu")
    console.print("[bold green]r[/bold green]: Repeat Context Menu")
    console.print("[bold green]q[/bold green]: Exit")
    console.print()
    
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        try:
            choice = console.input("[bold cyan]Select option: [/bold cyan]").lower().strip()
            
            if choice == 'm':
                console.print("[yellow]🔄 Returning to Main Menu...[/yellow]")
                # Import here to avoid circular import
                import importlib
                cli_module = importlib.import_module('pynucleus.cli')
                cli_module.show_interactive_menu()
                return
            elif choice == 'r':
                console.print("[yellow]🔄 Repeating Context Menu...[/yellow]")
                context_menu_func()
                return
            elif choice == 'q':
                console.print("[yellow]👋 Goodbye![/yellow]")
                sys.exit(0)
            else:
                console.print(f"[red]❌ Invalid choice: '{choice}'. Please enter 'm', 'r', or 'q'[/red]")
                attempts += 1
                
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            attempts += 1
    
    # If too many invalid attempts, default to main menu
    console.print("[yellow]⚠️  Too many invalid attempts. Returning to main menu.[/yellow]")
    # Import here to avoid circular import
    import importlib
    cli_module = importlib.import_module('pynucleus.cli')
    cli_module.show_interactive_menu()

@error_handler
def enhanced_context_menu(context_name: str, menu_options: list, command_executor: Callable[[str], None]):
    """
    Enhanced context menu with integrated post-command navigation.
    
    Args:
        context_name: Name of the context (e.g., "Ingest", "Health Check")
        menu_options: List of tuples (option_key, description, command)
        command_executor: Function to execute the selected command
    """
    console.print(f"\n[bold blue]🧪 PyNucleus - {context_name}[/bold blue]")
    console.print("Choose an option:\n")
    
    # Display menu options
    for option_key, description, _ in menu_options:
        console.print(f"[bold cyan]{option_key:>2}[/bold cyan]  {description}")
    
    # Add standard navigation options
    console.print(f"[bold cyan] m[/bold cyan]  Return to main menu")
    console.print(f"[bold cyan] q[/bold cyan]  Exit")
    console.print("\n" + "─" * 80)
    
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        try:
            choice = console.input(f"[bold green]Enter your choice: [/bold green]").strip().lower()
            
            if choice in ['q', 'quit']:
                console.print("👋 [yellow]Goodbye![/yellow]")
                sys.exit(0)
            elif choice in ['m', 'main']:
                console.print("[yellow]🔄 Returning to Main Menu...[/yellow]")
                # Import here to avoid circular import
                import importlib
                cli_module = importlib.import_module('pynucleus.cli')
                cli_module.show_interactive_menu()
                return
            
            # Check if it's a valid context command
            for option_key, description, command in menu_options:
                if choice == option_key:
                    console.print(f"\n[yellow]🔄 Executing: {description}[/yellow]")
                    console.print("─" * 80)
                    
                    # Execute the command
                    command_executor(command)
                    
                    # Show post-command options with current context
                    post_command_options(lambda: enhanced_context_menu(context_name, menu_options, command_executor))
                    return
            
            console.print(f"[red]❌ Invalid choice: '{choice}'. Please enter a valid option.[/red]")
            attempts += 1
                
        except (KeyboardInterrupt, EOFError):
            console.print("\n👋 [yellow]Goodbye![/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            attempts += 1
            
    # If we've exceeded max attempts, return to main menu
    console.print("[yellow]⚠️  Too many invalid attempts. Returning to main menu.[/yellow]")
    # Import here to avoid circular import
    import importlib
    cli_module = importlib.import_module('pynucleus.cli')
    cli_module.show_interactive_menu()

@error_handler
def simple_command_wrapper(command_name: str, command_executor: Callable[[], None]):
    """
    Wrapper for simple commands that don't have sub-menus.
    Executes the command and shows post-command options.
    
    Args:
        command_name: Name of the command for display
        command_executor: Function to execute the command
    """
    console.print(f"\n[yellow]🔄 Executing: {command_name}[/yellow]")
    console.print("─" * 80)
    
    try:
        # Execute the command
        command_executor()
        
        # Show post-command options - for simple commands, context menu returns to main
        def return_to_main():
            # Import here to avoid circular import
            import importlib
            cli_module = importlib.import_module('pynucleus.cli')
            cli_module.show_interactive_menu()
            
        post_command_options(return_to_main)
        
    except Exception as e:
        console.print(f"[red]❌ Command failed: {e}[/red]")
        # Still show post-command options even if command failed
        def return_to_main():
            # Import here to avoid circular import
            import importlib
            cli_module = importlib.import_module('pynucleus.cli')
            cli_module.show_interactive_menu()
            
        post_command_options(return_to_main) 