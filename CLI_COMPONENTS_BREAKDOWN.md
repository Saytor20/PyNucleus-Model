# PyNucleus CLI Components Breakdown

## Overview

The PyNucleus CLI is a comprehensive command-line interface built with **Typer** that provides access to the PyNucleus chemical process simulation and RAG system. The CLI supports both interactive menu-driven navigation and direct command execution.

## Architecture

### Core Structure
```
PyNucleus CLI
├── Main App (Typer)
├── Interactive Menu System
├── Main Commands (9 commands)
├── Sub-Apps (Typer sub-applications)
└── Helper Functions
```

## 1. Main Commands (Fully Implemented)

### 1.1 `run` - Execute Pipeline
**Purpose**: Execute the full PyNucleus pipeline (RAG + DWSIM + Export)

**What it does**:
- Loads configuration from JSON/CSV files
- Runs RAG analysis on chemical engineering documents
- Executes DWSIM chemical process simulations
- Exports comprehensive results to output directory

**Usage**:
```bash
# Basic usage
pynucleus run --config configs/production_config.json

# With custom output
pynucleus run --config configs/dev_config.json --output data/results

# Dry run (validate only)
pynucleus run --config configs/test_config.json --dry-run
```

**Implementation Status**: ✅ **Fully Implemented**
- Configuration validation
- Pipeline execution (mock implementation)
- Result export
- Error handling

### 1.2 `chat` - Interactive Chat
**Purpose**: Start interactive chat with PyNucleus RAG system

**Features**:
- Interactive Q&A with chemical engineering knowledge
- Single question mode with `--single` flag
- Configurable model parameters
- Pretty formatting for responses

**Usage**:
```bash
# Interactive mode
pynucleus chat

# Single question
pynucleus chat --single "What is distillation?"

# Custom model
pynucleus chat --model Qwen/Qwen2.5-1.5B-Instruct --temperature 0.8
```

**Implementation Status**: ✅ **Fully Implemented**
- RAG system integration
- Model loading and management
- Interactive session handling
- Response formatting

### 1.3 `build` - Plant Simulation
**Purpose**: Build chemical plant simulation using modular templates

**Features**:
- 22 African plant templates
- Comprehensive financial analysis
- Interactive parameter input
- Results export to JSON

**Usage**:
```bash
# Interactive mode
pynucleus build

# With parameters
pynucleus build --template 1 --feedstock natural_gas --capacity 1000
```

**Implementation Status**: ✅ **Fully Implemented**
- Template system
- Financial calculations
- Interactive prompts
- Results export

### 1.4 `system-status` - System Monitoring
**Purpose**: System status monitoring with comprehensive diagnostics

**Sub-commands**:
- `comprehensive`: Full resource, DB, vector-store, VENV check
- `validator`: Fast self-test (configs, models, env vars)

**Usage**:
```bash
# Comprehensive check
pynucleus system-status comprehensive

# Fast validation
pynucleus system-status validator
```

**Implementation Status**: ✅ **Fully Implemented**
- System health checks
- Configuration validation
- Performance metrics
- Live dashboard support

### 1.5 `version` - Version Information
**Purpose**: Show PyNucleus version and system information

**Implementation Status**: ✅ **Fully Implemented**

## 2. Menu-Driven Commands (Partially Implemented)

### 2.1 `ingest` - Document Ingestion
**Purpose**: Document ingestion and vector database management

**Current Issue**: Typer sub-app takes precedence over interactive menu

**Sub-commands** (via Typer sub-app):
- `auto`: Auto-detect and ingest documents
- `watch`: Watch directory for new files
- `single`: Ingest single file
- `info`: Show vector database information
- `clear`: Clear document database
- `validate`: Validate ingestion system

**Interactive Menu** (via main command):
- Shows numbered options with "Return to main menu" and "Exit"
- Delegates to sub-commands based on user choice

**Usage**:
```bash
# Interactive menu (from main menu)
pynucleus  # then select "5"

# Direct sub-command
pynucleus ingest auto
pynucleus ingest watch data/01_raw
```

**Implementation Status**: ⚠️ **Partially Implemented**
- ✅ Sub-commands implemented
- ✅ Interactive menu implemented
- ❌ Menu not shown when running `pynucleus ingest` directly

### 2.2 `eval` - Evaluation and Testing
**Purpose**: Run evaluations and compute metrics

**Sub-commands**:
- `golden`: Run golden dataset evaluation
- `metrics`: Compute RAG metrics

**Implementation Status**: ⚠️ **Partially Implemented**
- ✅ Sub-commands implemented
- ✅ Interactive menu implemented
- ❌ Menu not shown when running `pynucleus eval` directly

### 2.3 `serve` - Web Server Management
**Purpose**: Manage PyNucleus web server

**Sub-commands**:
- `start`: Start web server
- `stop`: Stop web server
- `restart`: Restart web server

**Implementation Status**: ⚠️ **Partially Implemented**
- ✅ Sub-commands implemented
- ✅ Interactive menu implemented
- ❌ Menu not shown when running `pynucleus serve` directly

### 2.4 `health` - System Health Checks
**Purpose**: Comprehensive system health diagnostics

**Sub-commands**:
- `quick`: Quick health check
- `full`: Full system diagnostics
- `network`: Network connectivity check
- `storage`: Storage system check

**Implementation Status**: ⚠️ **Partially Implemented**
- ✅ Sub-commands implemented
- ✅ Interactive menu implemented
- ❌ Menu not shown when running `pynucleus health` directly

## 3. Sub-Apps (Typer Applications)

### 3.1 `rag` - RAG System Operations
**Purpose**: Unified RAG system operations

**Status**: ✅ **Fully Implemented**
- Document vectorization
- VectorDB management
- Auto-ingest functionality
- Evaluation tools

### 3.2 Legacy Sub-Apps (To be removed)
- `ingest_app`: Will be replaced by main command
- `eval_app`: Will be replaced by main command
- `serve_app`: Will be replaced by main command
- `health_app`: Will be replaced by main command

## 4. Interactive Menu System

### 4.1 Main Menu
**Purpose**: Primary navigation interface

**Features**:
- Numbered options (1-9)
- Help option (0)
- Exit option (q)
- Tips and guidance

**Commands Available**:
1. Execute pipeline
2. Chat with LLM
3. Build plant simulation
4. System status
5. Ingest documents
6. Health check
7. Show version
8. Run evaluations
9. Web server
0. Show help
q. Exit

### 4.2 Context Menus
**Purpose**: Sub-menu navigation for complex commands

**Features**:
- Context-specific options
- Return to main menu option
- Exit option
- Post-command options

**Available Contexts**:
- Ingest options (6 sub-commands)
- Health options (4 sub-commands)
- Eval options (2 sub-commands)
- Serve options (3 sub-commands)
- System-status options (2 sub-commands)

### 4.3 Post-Command Options
**Purpose**: Navigation after command execution

**Options**:
- Return to main menu (m)
- Show context options again (r)
- Exit (q)

## 5. Error Handling

### 5.1 Error Decorator
**Purpose**: Consistent error handling across all commands

**Features**:
- Keyboard interrupt handling
- File not found errors
- Permission errors
- Unexpected errors
- Debug mode support

### 5.2 Logging System
**Purpose**: Structured logging for all CLI operations

**Features**:
- Configurable log levels
- File and console output
- Rich formatting
- Error tracking

## 6. Configuration Management

### 6.1 Settings Integration
**Purpose**: Access to PyNucleus settings

**Features**:
- Environment variable support
- Configuration file loading
- Default value handling
- Validation

### 6.2 Path Management
**Purpose**: Consistent path handling across commands

**Features**:
- Relative and absolute paths
- Path validation
- Directory creation
- File existence checks

## 7. Current Issues and Solutions

### 7.1 Issue: Menu Not Showing for Direct Commands
**Problem**: Running `pynucleus ingest` shows Typer help instead of interactive menu

**Root Cause**: Typer sub-app registration takes precedence over main command

**Solution**: Remove Typer sub-app registration for menu-driven commands
```python
# Remove these lines:
# app.add_typer(ingest_app, name="ingest")
# app.add_typer(eval_app)
# app.add_typer(serve_app)
# app.add_typer(health_app, name="health")
```

### 7.2 Issue: Command Delegation
**Problem**: Main commands need to delegate to sub-commands

**Solution**: Implement delegation logic in main command functions
```python
if subcommand:
    # Delegate to sub-command
    sys.argv = [sys.argv[0], 'ingest', subcommand] + sys.argv[3:]
    app()
else:
    # Show interactive menu
    show_interactive_menu()
```

## 8. Usage Patterns

### 8.1 Interactive Mode (Recommended)
```bash
# Start interactive session
./pynucleus

# Navigate through menus
# Select options with numbers
# Use 'm' to return to main menu
# Use 'q' to exit
```

### 8.2 Direct Command Mode
```bash
# Execute pipeline
./pynucleus run --config configs/production_config.json

# Chat with single question
./pynucleus chat --single "What is distillation?"

# Build plant
./pynucleus build --template 1 --feedstock natural_gas

# System status
./pynucleus system-status comprehensive
```

### 8.3 Sub-command Mode (After fix)
```bash
# Ingest documents
./pynucleus ingest  # Shows interactive menu
./pynucleus ingest auto  # Direct sub-command

# Health checks
./pynucleus health  # Shows interactive menu
./pynucleus health quick  # Direct sub-command
```

## 9. Future Enhancements

### 9.1 Planned Improvements
- [ ] Remove Typer sub-app conflicts
- [ ] Add command completion
- [ ] Implement command history
- [ ] Add configuration wizard
- [ ] Enhance error messages
- [ ] Add progress bars for long operations

### 9.2 Potential Features
- [ ] Batch command execution
- [ ] Command aliases
- [ ] Custom themes
- [ ] Export command history
- [ ] Integration with external tools

## 10. Testing

### 10.1 Manual Testing
```bash
# Test interactive menu
echo -e "5\nm\nq" | ./pynucleus

# Test direct commands
./pynucleus run --help
./pynucleus chat --help
./pynucleus build --help

# Test sub-commands
./pynucleus ingest auto --help
./pynucleus health quick --help
```

### 10.2 Automated Testing
- Unit tests for command functions
- Integration tests for menu system
- Error handling tests
- Configuration validation tests

## Summary

The PyNucleus CLI is a well-structured command-line interface with comprehensive functionality. The main issue is the conflict between Typer sub-apps and interactive menus for certain commands. Once this is resolved, users will have a seamless experience with both interactive navigation and direct command execution.

**Key Strengths**:
- Comprehensive command coverage
- Rich interactive menus
- Robust error handling
- Flexible usage patterns
- Good documentation

**Areas for Improvement**:
- Resolve sub-app conflicts
- Enhance user experience
- Add more automation features
- Improve error messages
- Add command completion 