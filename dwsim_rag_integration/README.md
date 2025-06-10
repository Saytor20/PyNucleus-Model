# Docker Configuration - Reorganized

The Docker configuration has been moved and consolidated into the new organized structure.

## 📁 New Location

All Docker configurations are now located in:
```
dwsim_rag_integration/config/
├── docker-compose.yml          # Service orchestration
├── Dockerfile.dwsim-service    # Enhanced DWSIM service container
└── README.md                   # Detailed documentation
```

## 🚀 Quick Start

To run the enhanced DWSIM service with RAG integration:

```bash
cd dwsim_rag_integration
docker-compose up --build
```

## 🧹 What Was Cleaned Up

The following duplicate and test files were removed:
- `mock_dwsim_service/` - Consolidated into `dwsim_rag_integration/service/`
- `dwsim_api/` - Merged with enhanced service
- `Dockerfile.mock` - Replaced with clean Dockerfile
- `nginx.conf` - Not needed for current architecture
- `Dockerfile.client` - Simplified deployment
- Multiple duplicate Dockerfiles

## 📖 Documentation

For complete documentation on the enhanced DWSIM+RAG integration, see:
- `dwsim_rag_integration/README.md` - Comprehensive setup and usage guide
- `dwsim_rag_integration/examples/` - Working demonstrations

---

**Note**: This folder is kept for backward compatibility. All new development should use the `dwsim_rag_integration` package. 