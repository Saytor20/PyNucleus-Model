"""
Minimal menu system for PyNucleus CLI - pipeline cleanup version
"""

from rich.console import Console

console = Console()

def enhanced_context_menu(title: str, menu_options: list, command_executor):
    """Enhanced context menu with navigation"""
    console.print(f"\n🔧 [bold blue]{title} Options[/bold blue]")
    
    if not menu_options:
        console.print("[yellow]No options available for this context[/yellow]")
        return 'main'
    
    # Display options
    for i, (option, description, command) in enumerate(menu_options, 1):
        console.print(f"[bold cyan]{i:>2}[/bold cyan]  {description}")
    
    console.print(f"[bold cyan] 0[/bold cyan]  Return to main menu")
    
    try:
        choice = input(f"\nEnter your choice (0-{len(menu_options)}): ").strip()
        
        if choice == '0' or choice.lower() in ['back', 'main', 'menu']:
            return 'main'
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(menu_options):
                option_key = menu_options[choice_num - 1][2]  # Use the command (third element)
                command_executor(option_key)
                return 'main'
            else:
                console.print(f"[red]❌ Invalid choice. Please select 0-{len(menu_options)}[/red]")
                return 'main'
        except ValueError:
            console.print("[red]❌ Please enter a valid number[/red]")
            return 'main'
            
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Returning to main menu...[/yellow]")
        return 'main'

def simple_command_wrapper(command_name: str, command_executor):
    """Simple wrapper for commands that don't need sub-menus"""
    try:
        console.print(f"\n🚀 [bold blue]Executing {command_name}...[/bold blue]")
        command_executor()
        
        # After command execution, offer navigation
        console.print("\n" + "─" * 60)
        console.print("🧭 [bold cyan]What would you like to do next?[/bold cyan]")
        console.print("   [cyan]1.[/cyan] Run another command")
        console.print("   [cyan]2.[/cyan] Return to main menu") 
        console.print("   [cyan]3.[/cyan] Exit PyNucleus")
        console.print("─" * 60)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            return 'main'  # Return to main menu to select another command
        elif choice == '2':
            return 'main'  # Return to main menu
        elif choice == '3':
            return 'quit'  # Exit the application
        else:
            console.print("[yellow]Invalid choice, returning to main menu...[/yellow]")
            return 'main'
            
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Returning to main menu...[/yellow]")
        return 'main'
    except Exception as e:
        console.print(f"[red]❌ Command failed: {e}[/red]")
        return 'main'