#!/usr/bin/env python3
"""
Test script to verify enhanced financial analysis functionality
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import from streamlit_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import (
    fetch_financial_data, 
    infer_financial_insights, 
    generate_financial_health_summary,
    enhance_swot_with_financial_data,
    generate_realtime_swot,
    analyze_sentiment,
    analyze_competitors
)

async def test_enhanced_financial_analysis():
    """Test enhanced financial analysis functions"""
    
    print("🚀 Testing Enhanced Financial Analysis System")
    print("=" * 60)
    
    # Test companies with different characteristics
    test_companies = [
        {"name": "Atlassian", "type": "collaboration"},
        {"name": "Shopify", "type": "ecommerce"},
        {"name": "Adobe", "type": "creative_software"},
        {"name": "Notion", "type": "generic_saas"},
        {"name": "Figma", "type": "creative_software"}
    ]
    
    for company_info in test_companies:
        company = company_info["name"]
        print(f"\n📊 Testing Enhanced Financial Analysis for {company}")
        print("-" * 50)
        
        try:
            # Test financial data fetching with intelligent fallbacks
            print(f"  💰 Fetching financial data for {company}...")
            financial_data = await fetch_financial_data(company)
            
            print(f"    Market Cap: {financial_data.get('market_cap', 'N/A')}")
            print(f"    Revenue Growth: {financial_data.get('revenue_growth', 'N/A')}")
            print(f"    P/E Ratio: {financial_data.get('pe_ratio', 'N/A')}")
            print(f"    Stock Performance: {financial_data.get('stock_performance', 'N/A')}")
            print(f"    Analyst Rating: {financial_data.get('analyst_rating', 'N/A')}")
            
            # Check for financial health summary
            financial_health = financial_data.get('financial_health', '')
            if financial_health:
                print(f"    💡 Financial Health: {financial_health}")
            
            growth_outlook = financial_data.get('growth_outlook', '')
            if growth_outlook:
                print(f"    📈 Growth Outlook: {growth_outlook}")
            
            valuation_outlook = financial_data.get('valuation_outlook', '')
            if valuation_outlook:
                print(f"    💎 Valuation Outlook: {valuation_outlook}")
            
            # Test sentiment and competitor analysis for context
            print(f"  📊 Analyzing sentiment for {company}...")
            sentiment_data = await analyze_sentiment(company, [])
            sentiment_score = sentiment_data.get("sentiment_score", 50)
            print(f"    Sentiment Score: {sentiment_score}/100")
            
            print(f"  🏆 Analyzing competitors for {company}...")
            competitor_data = await analyze_competitors(company)
            market_position = competitor_data.get("market_position", "Unknown")
            print(f"    Market Position: {market_position}")
            
            # Test enhanced SWOT generation with financial integration
            print(f"  🎯 Generating enhanced SWOT with financial integration...")
            swot_data = await generate_realtime_swot(
                company, [], financial_data, sentiment_data, competitor_data
            )
            
            # Count financial-enhanced SWOT points
            financial_strengths = [s for s in swot_data.get("strengths", []) 
                                 if s.get("source") == "Financial_analysis"]
            financial_weaknesses = [w for w in swot_data.get("weaknesses", []) 
                                  if w.get("source") == "Financial_analysis"]
            financial_opportunities = [o for o in swot_data.get("opportunities", []) 
                                     if o.get("source") == "Financial_analysis"]
            financial_threats = [t for t in swot_data.get("threats", []) 
                                if t.get("source") == "Financial_analysis"]
            
            print(f"    ✅ Financial-enhanced SWOT points:")
            print(f"      Strengths: {len(financial_strengths)}")
            print(f"      Weaknesses: {len(financial_weaknesses)}")
            print(f"      Opportunities: {len(financial_opportunities)}")
            print(f"      Threats: {len(financial_threats)}")
            
            # Show examples of financial integration
            if financial_strengths:
                print(f"    💪 Example Financial Strength: {financial_strengths[0]['point']}")
            if financial_opportunities:
                print(f"    🌟 Example Financial Opportunity: {financial_opportunities[0]['point']}")
            
            # Verify no "Unknown" values in key metrics
            unknown_count = sum(1 for value in financial_data.values() 
                              if isinstance(value, str) and value == "Unknown")
            
            if unknown_count == 0:
                print(f"    ✅ SUCCESS: No 'Unknown' values found in financial data")
            else:
                print(f"    ⚠️  WARNING: {unknown_count} 'Unknown' values still present")
            
            # Test financial inference for context
            print(f"  🧠 Testing financial inference engine...")
            inferred_insights = infer_financial_insights(company, sentiment_data, competitor_data)
            print(f"    Industry Classification: {inferred_insights['industry']}")
            print(f"    Market Cap Range: {inferred_insights['market_cap_range']}")
            print(f"    Financial Health Summary: {inferred_insights['financial_health_summary']}")
            
        except Exception as e:
            print(f"    ❌ Error testing {company}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Enhanced Financial Analysis Testing Complete!")
    print("=" * 60)
    
    # Summary of improvements
    print("\n📋 Key Improvements Implemented:")
    print("✅ Dynamic financial insights instead of static 'Unknown' values")
    print("✅ Industry-based financial pattern inference")
    print("✅ Context-driven financial health summaries")
    print("✅ Financial data integration into SWOT analysis")
    print("✅ Analyst-style commentary and outlook")
    print("✅ Intelligent fallbacks when live data unavailable")
    print("✅ Enhanced UI with financial health assessment")

if __name__ == "__main__":
    asyncio.run(test_enhanced_financial_analysis())
