#!/bin/bash

API_URL="http://localhost:8000"

echo "Testing /health endpoint..."
curl -s "$API_URL/health" | python -m json.tool

echo ""
echo "Testing /predict endpoint..."
python -c "import json; print(json.dumps({'data': [0.0]*784}))" | \
curl -s -X POST "$API_URL/predict" \
     -H "Content-Type: application/json" \
     -d @- | python -m json.tool

echo ""
echo "Testing /predict/batch endpoint..."
python -c "import json; print(json.dumps({'data': [[0.0]*784, [0.0]*784]}))" | \
curl -s -X POST "$API_URL/predict/batch" \
     -H "Content-Type: application/json" \
     -d @- | python -m json.tool

echo ""
echo "Tests completed!"