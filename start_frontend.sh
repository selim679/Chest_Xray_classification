#!/bin/bash
echo "=========================================="
echo " PneumoScan AI — Starting Angular Frontend"
echo "=========================================="

cd "$(dirname "$0")/frontend-angular"

if [ ! -d "node_modules" ]; then
    echo "Installing Node packages (first run, takes ~2 min)..."
    npm install
fi

echo ""
echo "🌐 Starting Angular on http://localhost:4200"
echo ""
npm start
