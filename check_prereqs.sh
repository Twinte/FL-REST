#!/bin/bash

echo "------ CHECKING PREREQUISITES ------"

# 1. Check Git
if command -v git &> /dev/null; then
    echo "✅ Git is installed: $(git --version)"
else
    echo "❌ Git is MISSING"
fi

# 2. Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed: $(docker --version)"
    
    # Check if user is in docker group
    if groups $USER | grep &>/dev/null 'docker'; then
        echo "   (User is correctly in the 'docker' group)"
    else
        echo "   ⚠️ WARNING: User is NOT in 'docker' group (you will need sudo for every command)"
    fi
else
    echo "❌ Docker is MISSING"
fi

# 3. Check Docker Compose
if docker compose version &> /dev/null; then
    echo "✅ Docker Compose is installed: $(docker compose version)"
else
    echo "❌ Docker Compose is MISSING"
fi

# 4. Check NVIDIA Container Toolkit (For GPU Support)
if docker info | grep -i "CDI" &> /dev/null || docker info | grep -i "nvidia" &> /dev/null; then
    echo "✅ NVIDIA Container Toolkit seems active in Docker."
else
    echo "⚠️ NVIDIA Container Toolkit not detected in Docker (GPU clients will fall back to CPU)."
fi

echo "------------------------------------"