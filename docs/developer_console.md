# PyNucleus Developer Console

## Overview

The PyNucleus Developer Console is a retro-styled CRT terminal interface that provides developers with direct access to the system's core functionality for testing and diagnostics.

## Access

Once the PyNucleus web application is running:

```bash
python run_web_app.py
```

Navigate to: **http://localhost:5001/dev**

## Features

### 🔍 Ask the Model
- **Interactive RAG Testing**: Submit questions directly to the PyNucleus RAG system
- **Real-time Processing**: See live answers with metadata including processing time, source count, and model information
- **Formatted Output**: Clean display of questions, answers, sources, and system metadata

**Keyboard Shortcuts:**
- `Ctrl+Enter`: Submit question from text area
- `F5`: Run system diagnostics
- `F6`: Run comprehensive diagnostics
- `F7`: Load system statistics

### 📁 Document Upload
- **Direct Upload**: Upload documents directly to the source_documents folder
- **Auto-Processing**: Documents are automatically processed by DocumentProcessor
- **PDF Table Extraction**: Automatic table extraction from PDF files using Camelot
- **Multiple Formats**: Support for TXT, PDF, MD, DOC, and DOCX files
- **Real-time Feedback**: See upload status and processing results immediately

### 🔧 System Diagnostics
- **System Validator**: Run focused validation tests for accuracy and citations
- **Comprehensive Diagnostics**: Run full system health checks and component validation
- **System Statistics**: View detailed database, document, and system information
- **Performance Metrics**: View response times, success rates, and system status
- **Component Health**: Monitor ChromaDB, Qwen models, and PDF processing systems
- **Real-time Output**: See diagnostic results in formatted terminal display

### 📊 Statistics Dashboard
- **Vector Database Stats**: Document count, database size, and health status
- **Document Statistics**: Source document count, file types, and processing status
- **System Information**: Platform details, memory usage, and CPU metrics
- **Storage Analysis**: Data directory size and structure information

## Visual Design

The developer console features a classic monochrome CRT terminal aesthetic:
- **Green-on-black color scheme** (`#33ff00` on `#0d0d0d`)
- **Courier New monospace font** for that authentic terminal feel
- **CRT scan lines effect** with subtle overlay patterns
- **Retro box-drawing characters** for structured output
- **Responsive design** that works on both desktop and mobile

## API Endpoints

The console interfaces with these backend endpoints:

### POST /ask
Accepts both JSON and form data:
```bash
# JSON format
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is distillation?"}' \
  http://localhost:5001/ask

# Form data format (used by HTMX)
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What is distillation?" \
  http://localhost:5001/ask
```

### GET /system_diagnostic
Returns system validator output as plain text:
```bash
curl http://localhost:5001/system_diagnostic
```

### GET /comprehensive_diagnostic
Returns comprehensive system diagnostic output as plain text:
```bash
curl http://localhost:5001/comprehensive_diagnostic
```

### GET /system_statistics
Returns detailed system statistics as JSON:
```bash
curl http://localhost:5001/system_statistics
```

### POST /upload
Upload documents to source_documents folder:
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5001/upload
```

## Technical Implementation

### Frontend
- **HTMX**: Provides seamless AJAX interactions without JavaScript frameworks
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **Custom CSS**: CRT effects and retro styling
- **Progressive Enhancement**: Works with JavaScript disabled

### Backend
- **Flask Integration**: Seamlessly integrated with existing PyNucleus API
- **Dual Input Support**: Handles both JSON and form-encoded requests
- **Error Handling**: Graceful degradation with informative error messages
- **Structured Logging**: All interactions logged for debugging

### System Validator Integration
- **JSON Output Mode**: New `--json` flag for machine-readable results
- **Real-time Execution**: Diagnostics run on-demand
- **Fallback Support**: Multiple levels of error handling and recovery

## Development

### Testing
Run the test suite to validate console functionality:
```bash
python -m pytest tests/test_dev_console.py -v
```

### Customization
Key files for modifications:
- `src/pynucleus/api/static/developer_dashboard.html` - Frontend interface
- `src/pynucleus/api/app.py` - Backend endpoints (`/ask`, `/dev`, `/system_diagnostic`)
- `scripts/system_validator.py` - Diagnostic system with `--json` output

## Usage Examples

### Basic Question Testing
1. Navigate to http://localhost:5001/dev
2. Type question: "What are the main advantages of modular chemical plants?"
3. Click "Run" or press `Ctrl+Enter`
4. Review formatted answer with sources and metadata

### System Health Check
1. Click "Run Diagnostics" or press `F5`
2. Wait for validation to complete
3. Review structured health report with test results
4. Check success rate and validation health status

### Continuous Development Workflow
1. Make changes to PyNucleus code
2. Use developer console to quickly test changes
3. Run diagnostics to ensure system health
4. Iterate based on real-time feedback

## Troubleshooting

### Console Not Loading
- Ensure Flask app is running on port 5001
- Check browser console for JavaScript errors
- Verify HTMX CDN is accessible

### Ask Function Not Working
- Check that the RAG system is properly initialized
- Verify ChromaDB is accessible and contains documents
- Review Flask logs for detailed error messages

### Diagnostics Timing Out
- System validation can take 30-60 seconds
- Check that all required dependencies are installed
- Monitor system resources during validation

## Integration with Existing Workflow

The developer console integrates seamlessly with existing PyNucleus development workflows:

- **Local Development**: Quick testing without external tools
- **CI/CD Integration**: Diagnostic endpoints can be called programmatically
- **Documentation**: Live examples for API behavior
- **Debugging**: Real-time system state inspection

For production deployments, consider restricting access to the `/dev` endpoint or disabling it entirely for security. 