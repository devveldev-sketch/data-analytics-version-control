#!/usr/bin/env python3
"""
Quick test script to verify SWOT analysis functionality
"""
import asyncio
import json
from streamlit_app import run_analyze_with_trace, get_default_swot

async def test_adobe_analysis():
    """Test Adobe SWOT analysis"""
    print("Testing Adobe SWOT analysis...")
    
    try:
        result = await run_analyze_with_trace("Adobe", lookback_days=30, top_k=10)
        
        if result:
            print("✅ Analysis completed successfully")
            
            parsed = result.get("json", {})
            company_name = parsed.get("company", "Unknown")
            print(f"Company: {company_name}")
            print(f"Company type: {type(company_name)}")
            if isinstance(company_name, dict):
                print(f"Company dict keys: {list(company_name.keys())}")
            
            swot = parsed.get("swot", {})
            for category in ["strengths", "weaknesses", "opportunities", "threats"]:
                items = swot.get(category, [])
                print(f"\n{category.upper()} ({len(items)} items):")
                for i, item in enumerate(items[:3], 1):  # Show first 3 items
                    if isinstance(item, dict):
                        point = item.get("point", str(item))
                    else:
                        point = str(item)
                    print(f"  {i}. {point}")
                    
            print(f"\nGenerated text preview: {result.get('markdown', '')[:200]}...")
            
        else:
            print("❌ Analysis failed - no result returned")
            
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")

def test_default_swot():
    """Test default SWOT generation"""
    print("\nTesting default SWOT for Adobe...")
    
    default_swot = get_default_swot("Adobe")
    
    for category in ["strengths", "weaknesses", "opportunities", "threats"]:
        items = default_swot.get(category, [])
        print(f"\n{category.upper()} ({len(items)} items):")
        for i, item in enumerate(items[:2], 1):  # Show first 2 items
            if isinstance(item, dict):
                point = item.get("point", str(item))
            else:
                point = str(item)
            print(f"  {i}. {point}")

if __name__ == "__main__":
    print("🔍 Testing SWOT Analysis Functionality\n")
    
    # Test default SWOT first (doesn't require API calls)
    test_default_swot()
    
    # Test full analysis (requires API calls)
    print("\n" + "="*50)
    asyncio.run(test_adobe_analysis())
    
    print("\n✅ Test completed!")
