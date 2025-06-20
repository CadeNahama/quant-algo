#!/usr/bin/env python3
"""
Report Generator Module for Quantitative Trading System

This module generates comprehensive reports from backtest results and trading data,
providing insights and analysis in a readable format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    """
    Generates comprehensive trading reports from backtest results and market data.
    """
    
    def __init__(self):
        self.report_data = {}
        
    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance metrics and generate insights.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Extract key metrics
        total_return = results.get('total_return', 0)
        annualized_return = results.get('annualized_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        volatility = results.get('volatility', 0)
        win_rate = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)
        
        # Performance classification
        if annualized_return > 0.15:
            performance_rating = "Excellent"
        elif annualized_return > 0.08:
            performance_rating = "Good"
        elif annualized_return > 0.02:
            performance_rating = "Moderate"
        elif annualized_return > -0.05:
            performance_rating = "Poor"
        else:
            performance_rating = "Very Poor"
            
        # Risk assessment
        if sharpe_ratio > 1.5:
            risk_rating = "Low Risk"
        elif sharpe_ratio > 0.8:
            risk_rating = "Moderate Risk"
        elif sharpe_ratio > 0.3:
            risk_rating = "High Risk"
        else:
            risk_rating = "Very High Risk"
            
        # Drawdown assessment
        if max_drawdown < 0.05:
            drawdown_rating = "Excellent"
        elif max_drawdown < 0.10:
            drawdown_rating = "Good"
        elif max_drawdown < 0.20:
            drawdown_rating = "Moderate"
        else:
            drawdown_rating = "Poor"
            
        analysis.update({
            'performance_rating': performance_rating,
            'risk_rating': risk_rating,
            'drawdown_rating': drawdown_rating,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': total_trades
        })
        
        return analysis
    
    def generate_market_insights(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate market insights from the collected data.
        
        Args:
            data_dict: Dictionary of stock data
            
        Returns:
            Dictionary with market insights
        """
        insights = {}
        
        # Calculate market statistics
        all_returns = []
        all_volumes = []
        symbols = []
        
        for symbol, data in data_dict.items():
            if not data.empty:
                returns = data['returns'].dropna()
                volumes = data['Volume'].dropna()
                
                all_returns.extend(returns.tolist())
                all_volumes.extend(volumes.tolist())
                symbols.append(symbol)
        
        if all_returns:
            # Market volatility
            market_volatility = np.std(all_returns) * np.sqrt(252)
            
            # Best and worst performers
            symbol_performances = {}
            for symbol, data in data_dict.items():
                if not data.empty and len(data) > 50:
                    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
                    symbol_performances[symbol] = total_return
            
            if symbol_performances:
                best_performer = max(symbol_performances.items(), key=lambda x: x[1])
                worst_performer = min(symbol_performances.items(), key=lambda x: x[1])
                
                insights.update({
                    'market_volatility': market_volatility,
                    'best_performer': best_performer,
                    'worst_performer': worst_performer,
                    'num_symbols': len(symbols),
                    'avg_daily_volume': np.mean(all_volumes) if all_volumes else 0
                })
        
        return insights
    
    def generate_strategy_insights(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights about the strategy performance.
        
        Args:
            signals_df: DataFrame with momentum signals
            
        Returns:
            Dictionary with strategy insights
        """
        insights = {}
        
        if not signals_df.empty:
            # Signal strength analysis
            momentum_columns = [col for col in signals_df.columns if 'momentum' in col.lower()]
            
            if momentum_columns:
                try:
                    # Filter out non-numeric columns and calculate statistics
                    numeric_momentum = signals_df[momentum_columns].select_dtypes(include=[np.number])
                    if not numeric_momentum.empty:
                        # Use numpy to calculate overall statistics
                        all_values = numeric_momentum.values.flatten()
                        avg_momentum = np.mean(all_values)
                        momentum_volatility = np.std(all_values)
                        
                        insights.update({
                            'avg_momentum': avg_momentum,
                            'momentum_volatility': momentum_volatility,
                            'signal_strength': 'Strong' if abs(avg_momentum) > 0.1 else 'Moderate' if abs(avg_momentum) > 0.05 else 'Weak'
                        })
                except Exception as e:
                    # Fallback values if calculation fails
                    insights.update({
                        'avg_momentum': 0.0,
                        'momentum_volatility': 0.0,
                        'signal_strength': 'Unknown'
                    })
        
        return insights
    
    def generate_executive_summary(self, 
                                 performance_analysis: Dict[str, Any],
                                 market_insights: Dict[str, Any],
                                 strategy_insights: Dict[str, Any]) -> str:
        """
        Generate a 1-2 paragraph executive summary.
        
        Args:
            performance_analysis: Performance analysis results
            market_insights: Market insights
            strategy_insights: Strategy insights
            
        Returns:
            Executive summary as a string
        """
        # Extract key metrics
        annualized_return = performance_analysis.get('annualized_return', 0)
        sharpe_ratio = performance_analysis.get('sharpe_ratio', 0)
        max_drawdown = performance_analysis.get('max_drawdown', 0)
        performance_rating = performance_analysis.get('performance_rating', 'Unknown')
        risk_rating = performance_analysis.get('risk_rating', 'Unknown')
        
        best_performer = market_insights.get('best_performer', ('Unknown', 0))
        worst_performer = market_insights.get('worst_performer', ('Unknown', 0))
        market_volatility = market_insights.get('market_volatility', 0)
        
        signal_strength = strategy_insights.get('signal_strength', 'Unknown')
        
        # Generate summary
        summary = f"""
EXECUTIVE SUMMARY - MOMENTUM TRADING STRATEGY ANALYSIS

The momentum trading strategy demonstrated {performance_rating.lower()} performance with an annualized return of {annualized_return:.2%} and a Sharpe ratio of {sharpe_ratio:.2f}, indicating {risk_rating.lower()} characteristics. The strategy experienced a maximum drawdown of {max_drawdown:.2%}, which is considered {performance_analysis.get('drawdown_rating', 'Unknown').lower()} risk management. During the testing period, the strategy executed {performance_analysis.get('total_trades', 0)} trades with a win rate of {performance_analysis.get('win_rate', 0):.1%}.

Market analysis reveals that {best_performer[0]} was the top performer with a {best_performer[1]:.2%} return, while {worst_performer[0]} underperformed with a {worst_performer[1]:.2%} return. The overall market volatility was {market_volatility:.2%} annually, and the momentum signals showed {signal_strength.lower()} strength. The strategy's performance suggests that momentum factors are {'effective' if annualized_return > 0.05 else 'challenging'} in the current market environment, with {'favorable' if sharpe_ratio > 0.5 else 'unfavorable'} risk-adjusted returns.
        """
        
        return summary.strip()
    
    def generate_detailed_report(self, 
                               results: Dict[str, Any],
                               data_dict: Dict[str, pd.DataFrame],
                               signals_df: pd.DataFrame) -> str:
        """
        Generate a comprehensive detailed report.
        
        Args:
            results: Backtest results
            data_dict: Stock data dictionary
            signals_df: Momentum signals DataFrame
            
        Returns:
            Complete report as a string
        """
        # Generate analyses
        performance_analysis = self.analyze_performance(results)
        market_insights = self.generate_market_insights(data_dict)
        strategy_insights = self.generate_strategy_insights(signals_df)
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(
            performance_analysis, market_insights, strategy_insights
        )
        
        # Generate detailed report
        report = f"""
{executive_summary}

DETAILED PERFORMANCE ANALYSIS
============================

PERFORMANCE METRICS:
• Total Return: {performance_analysis.get('total_return', 0):.2%}
• Annualized Return: {performance_analysis.get('annualized_return', 0):.2%}
• Sharpe Ratio: {performance_analysis.get('sharpe_ratio', 0):.2f}
• Maximum Drawdown: {performance_analysis.get('max_drawdown', 0):.2%}
• Volatility: {performance_analysis.get('volatility', 0):.2%}
• Win Rate: {performance_analysis.get('win_rate', 0):.1%}
• Total Trades: {performance_analysis.get('total_trades', 0)}

MARKET INSIGHTS:
• Market Volatility: {market_insights.get('market_volatility', 0):.2%}
• Best Performer: {market_insights.get('best_performer', ('N/A', 0))[0]} ({market_insights.get('best_performer', ('N/A', 0))[1]:.2%})
• Worst Performer: {market_insights.get('worst_performer', ('N/A', 0))[0]} ({market_insights.get('worst_performer', ('N/A', 0))[1]:.2%})
• Number of Symbols: {market_insights.get('num_symbols', 0)}
• Average Daily Volume: {market_insights.get('avg_daily_volume', 0):,.0f}

STRATEGY INSIGHTS:
• Average Momentum: {strategy_insights.get('avg_momentum', 0):.4f}
• Momentum Volatility: {strategy_insights.get('momentum_volatility', 0):.4f}
• Signal Strength: {strategy_insights.get('signal_strength', 'N/A')}

RECOMMENDATIONS:
{self._generate_recommendations(performance_analysis, market_insights, strategy_insights)}

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report.strip()
    
    def _generate_recommendations(self, 
                                performance_analysis: Dict[str, Any],
                                market_insights: Dict[str, Any],
                                strategy_insights: Dict[str, Any]) -> str:
        """
        Generate recommendations based on analysis.
        
        Args:
            performance_analysis: Performance analysis
            market_insights: Market insights
            strategy_insights: Strategy insights
            
        Returns:
            Recommendations as a string
        """
        recommendations = []
        
        # Performance-based recommendations
        annualized_return = performance_analysis.get('annualized_return', 0)
        sharpe_ratio = performance_analysis.get('sharpe_ratio', 0)
        max_drawdown = performance_analysis.get('max_drawdown', 0)
        win_rate = performance_analysis.get('win_rate', 0)
        
        # 1. Adjust Momentum Thresholds
        if annualized_return < 0.05 or win_rate < 0.4:
            recommendations.append("• ADJUST MOMENTUM THRESHOLDS: Try loosening your criteria for when to buy/sell—your signals were likely too strict")
            recommendations.append("  → Consider reducing momentum thresholds from current levels to capture more trading opportunities")
            recommendations.append("  → Implement dynamic thresholds that adapt to market volatility")
        
        # 2. Review Position Sizing
        if max_drawdown > 0.15 or sharpe_ratio < 0.5:
            recommendations.append("• REVIEW POSITION SIZING: Ensure you're not risking too much (or too little) on each trade")
            recommendations.append("  → Implement Kelly Criterion or risk-parity based position sizing")
            recommendations.append("  → Consider reducing position sizes during high volatility periods")
        
        # 3. Improve Risk Management
        if sharpe_ratio < 0.5 or max_drawdown > 0.10:
            recommendations.append("• IMPROVE RISK MANAGEMENT: Add stop-losses, volatility filters, or correlation controls")
            recommendations.append("  → Implement trailing stop-losses at 2-3% below entry prices")
            recommendations.append("  → Add volatility filters to avoid trading during extreme market conditions")
            recommendations.append("  → Monitor correlation between positions and limit sector concentration")
        
        # 4. Use Multiple Timeframes
        if annualized_return < 0.08:
            recommendations.append("• USE MULTIPLE TIMEFRAMES: Combining short-, mid-, and long-term momentum might help")
            recommendations.append("  → Combine 20-day, 60-day, and 120-day momentum signals")
            recommendations.append("  → Weight signals based on their historical effectiveness")
            recommendations.append("  → Use shorter timeframes for entry/exit and longer for trend confirmation")
        
        # 5. Defensive Positioning
        market_volatility = market_insights.get('market_volatility', 0)
        if market_volatility > 0.25:
            recommendations.append("• DEFENSIVE POSITIONING: In volatile markets, consider reducing exposure or rotating into less risky sectors")
            recommendations.append("  → Reduce overall portfolio exposure during high volatility periods")
            recommendations.append("  → Rotate into defensive sectors (utilities, consumer staples) during market stress")
            recommendations.append("  → Increase cash allocation when volatility exceeds historical averages")
        
        # 6. Try Alternative Factors
        signal_strength = strategy_insights.get('signal_strength', 'Unknown')
        if signal_strength == 'Weak' or annualized_return < 0.05:
            recommendations.append("• TRY ALTERNATIVE FACTORS: Momentum may not be working right now. Try combining it with value, quality, or low volatility factors")
            recommendations.append("  → Combine momentum with value factors (P/E, P/B ratios)")
            recommendations.append("  → Add quality metrics (ROE, debt-to-equity, earnings stability)")
            recommendations.append("  → Include low volatility stocks to reduce portfolio risk")
            recommendations.append("  → Consider mean reversion strategies as a complement to momentum")
        
        # 7. Recheck Lookback Periods
        if signal_strength == 'Weak':
            recommendations.append("• RECHECK LOOKBACK PERIODS: The period over which you calculate momentum might be too short/long for this market cycle")
            recommendations.append("  → Test different lookback periods (10, 20, 50, 100 days)")
            recommendations.append("  → Use adaptive lookback periods based on market regime")
            recommendations.append("  → Consider using exponential moving averages instead of simple moving averages")
        
        # Additional specific recommendations based on performance
        if annualized_return > 0.10 and sharpe_ratio > 1.0:
            recommendations.append("• STRATEGY SHOWS PROMISE: Consider scaling up gradually while monitoring for regime changes")
            recommendations.append("  → Increase position sizes incrementally")
            recommendations.append("  → Monitor for changes in market conditions that may affect momentum effectiveness")
        
        if total_trades := performance_analysis.get('total_trades', 0):
            if total_trades < 10:
                recommendations.append("• LOW TRADING ACTIVITY: The strategy may be too conservative")
                recommendations.append("  → Review entry/exit criteria to increase trading frequency")
                recommendations.append("  → Consider expanding the universe of stocks analyzed")
        
        # Market-specific recommendations
        best_performer = market_insights.get('best_performer', ('Unknown', 0))
        worst_performer = market_insights.get('worst_performer', ('Unknown', 0))
        
        if best_performer[1] > 0.20:
            recommendations.append(f"• STRONG PERFORMER DETECTED: {best_performer[0]} showed exceptional performance ({best_performer[1]:.1%})")
            recommendations.append("  → Analyze what factors contributed to this outperformance")
            recommendations.append("  → Consider sector rotation strategies based on strong performers")
        
        if worst_performer[1] < -0.10:
            recommendations.append(f"• WEAK PERFORMER IDENTIFIED: {worst_performer[0]} underperformed significantly ({worst_performer[1]:.1%})")
            recommendations.append("  → Review risk management for similar stocks")
            recommendations.append("  → Consider short-selling opportunities or avoiding similar stocks")
        
        if not recommendations:
            recommendations.append("• CONTINUE MONITORING: Strategy performance is within acceptable parameters")
            recommendations.append("  → Maintain current risk management framework")
            recommendations.append("  → Continue regular performance reviews")
        
        return "\n".join(recommendations)
    
    def save_report(self, report: str, filename: str = "trading_report.txt"):
        """
        Save the report to a file.
        
        Args:
            report: Report content
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {filename}")

def main():
    """Example usage of the ReportGenerator."""
    generator = ReportGenerator()
    
    # Example data (you would replace this with actual results)
    example_results = {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'volatility': 0.18,
        'win_rate': 0.65,
        'total_trades': 45
    }
    
    # Generate and save report
    report = generator.generate_detailed_report(
        example_results, {}, pd.DataFrame()
    )
    generator.save_report(report)
    
    print("✅ Report generation completed!")

if __name__ == "__main__":
    main() 