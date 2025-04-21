import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from operator import itemgetter

def get_top_messages_by_period(message_info, period_days=90, top_n=5):
    """Get top N messages for each period."""
    if not message_info:
        return []
    
    # Get date range
    start_date = min(msg['date'] for msg in message_info)
    end_date = max(msg['date'] for msg in message_info)
    
    # Calculate number of periods
    total_days = (end_date - start_date).days
    periods = []
    
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=period_days), end_date)
        
        # Get messages in this period
        period_messages = [
            msg for msg in message_info 
            if current_start <= msg['date'] < current_end
        ]
        
        if period_messages:
            # Sort messages by engagement and get top N
            top_messages = sorted(period_messages, 
                                key=lambda x: x['engagement'], 
                                reverse=True)[:top_n]
            
            periods.append({
                'period_start': current_start,
                'period_end': current_end,
                'top_messages': top_messages
            })
        
        current_start = current_end
    
    return periods

def visualize_likeflation(filepath):
    """
    Visualize the number of likes (favorites) over time with trend line and growth statistics.
    """
    # Read messages
    with open(filepath, "r") as f:
        messages = json.load(f)
    
    # Sort messages by timestamp to ensure chronological order
    messages.sort(key=lambda x: x['created_at'])
    
    # Extract timestamps and like counts, and store message info
    timestamps = []
    like_counts = []
    message_info = []
    
    for message in messages:
        timestamp = message['created_at']
        likes = len(message['favorited_by']) if 'favorited_by' in message else 0
        reaction_count = sum(len(reaction['user_ids']) for reaction in message['reactions']) if 'reactions' in message else 0
        total_engagement = likes + reaction_count
        
        timestamps.append(timestamp)
        like_counts.append(total_engagement)
        
        message_info.append({
            'text': message['text'],
            'engagement': total_engagement,
            'date': datetime.fromtimestamp(timestamp),
            'name': message['name']
        })
    
    # Get top messages by period
    period_data = get_top_messages_by_period(message_info)
    
    print("\nTop 5 Messages by 90-Day Period:")
    print("=" * 80)
    for i, period in enumerate(period_data, 1):
        print(f"\nPeriod {i}: {period['period_start'].strftime('%Y-%m-%d')} to {period['period_end'].strftime('%Y-%m-%d')}")
        print("-" * 80)
        
        for j, message in enumerate(period['top_messages'], 1):
            print(f"\n#{j}")
            print(f"Author: {message['name']}")
            print(f"Date: {message['date'].strftime('%Y-%m-%d')}")
            print(f"Engagement: {message['engagement']}")
            print(f"Message: {message['text']}")
            if j < len(period['top_messages']):
                print("-" * 40)  # Separator between messages
        
        print("=" * 80)  # Separator between periods
    
    # Convert timestamps to datetime for plotting
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    first_date = dates[0]
    last_date = dates[-1]
    
    # Convert to numpy arrays for fitting
    x = np.array([(d - min(dates)).days for d in dates])
    y = np.array(like_counts)
    
    # Fit a polynomial (degree=1 for linear fit)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Get first and last points on the line of best fit
    first_point = p(x[0])
    last_point = p(x[-1])
    engagement_difference = last_point - first_point
    
    print("\nLine of Best Fit Analysis:")
    print("-" * 80)
    print(f"Date range: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"Starting engagement (trend line): {first_point:.2f}")
    print(f"Ending engagement (trend line): {last_point:.2f}")
    print(f"Total change in engagement: {engagement_difference:.2f}")
    print(f"Percentage change: {(engagement_difference/first_point * 100):.1f}%")
    print("-" * 80)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot scatter of actual data
    plt.scatter(dates, like_counts, alpha=0.5, label='Actual Engagement')
    
    # Plot trend line
    plt.plot(dates, p(x), "r--", label=f'Trend Line (slope: {z[0]:.3f})')
    
    # Highlight first and last points on trend line
    plt.scatter([dates[0], dates[-1]], [first_point, last_point], 
                color='red', s=100, zorder=5, 
                label='Trend Line Endpoints')
    
    # Plot top messages from each period
    period_dates = [p['top_messages'][0]['date'] for p in period_data]
    period_engagements = [p['top_messages'][0]['engagement'] for p in period_data]
    plt.scatter(period_dates, period_engagements, color='yellow', edgecolor='black', 
                s=150, zorder=6, label='Period Top Messages')
    
    # Calculate moving average
    window = 30  # 30-day moving average
    moving_avg = []
    for i in range(len(like_counts)):
        start = max(0, i - window)
        moving_avg.append(np.mean(like_counts[start:i+1]))
    
    # Plot moving average
    plt.plot(dates, moving_avg, 'g-', label=f'{window}-Day Moving Average', alpha=0.7)
    
    # Customize the plot
    plt.title('Message Engagement Over Time', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Engagement (Likes + Reactions)', fontsize=12)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add text box with trend line statistics
    stats_text = f'Date Range:\n{first_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")}\n\n'
    stats_text += f'Starting Engagement: {first_point:.1f}\n'
    stats_text += f'Ending Engagement: {last_point:.1f}\n'
    stats_text += f'Total Change: {engagement_difference:+.1f}\n'
    stats_text += f'Percent Change: {(engagement_difference/first_point * 100):+.1f}%'
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    return plt.gcf()

# Example usage:
if __name__ == "__main__":
    filepath = "training.json"
    fig = visualize_likeflation(filepath)
    plt.show()