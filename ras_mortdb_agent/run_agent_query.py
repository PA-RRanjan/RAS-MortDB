import asyncio
from fastmcp import Client
from mcp_server import mcp

async def test_agent():
    async with Client(mcp) as client:
        print("\n--- 1. Testing Deployment Recommendation ---")
        rec = await client.call_tool("recommend_deployment", {
            "target_hardware": "raspberry_pi_or_cpu_edge",
            "annotated_training_images_available": 1500,
            "priority": "fastest_edge_inference"
        })
        print(rec.structured_content or rec.content[0].text)

        print("\n--- 2. Testing Mortality Detection via ONNX ---")
        det = await client.call_tool("detect_mortality", {
            "image_path": "test_images/tl_0031_0153_20221022_001226_jpg.rf.5a6ee26f647296aab6267d96b17ced6f.jpg",
            "model": "yolov8n",
            "weight_format": "onnx",
            "conf": 0.25
        })
        print(det.structured_content or det.content[0].text)

asyncio.run(test_agent())
