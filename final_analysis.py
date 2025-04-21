import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import os

def run_analysis(year_1_messages_filename, year_1_analysis_filename, 
                 year_2_messages_filename, year_2_analysis_filename,
                 output_dir="rankings_output"):
    """
    Generate comprehensive rankings from GroupMe data across two time periods.
    
    Args:
        year_1_messages_filename: Path to first year's messages JSON
        year_1_analysis_filename: Path to first year's analysis JSON
        year_2_messages_filename: Path to second year's messages JSON
        year_2_analysis_filename: Path to second year's analysis JSON
        output_dir: Directory to save ranking results
    """
    print("Loading data files...")
    
    # Load all data files
    messages_1 = []
    messages_2 = []
    analysis_1 = {}
    analysis_2 = {}
    
    with open(year_1_messages_filename, 'r') as f:
        messages_1 = json.load(f)
    with open(year_2_messages_filename, 'r') as f:
        messages_2 = json.load(f)
    with open(year_1_analysis_filename, 'r') as f:
        analysis_1 = json.load(f)
    with open(year_2_analysis_filename, 'r') as f:
        analysis_2 = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing data for {len(analysis_1['users'])} users in year 1 and {len(analysis_2['users'])} users in year 2...")
    
    # Generate all rankings
    results = {}
    
    # Single-year rankings
    print("Generating year 1 rankings...")
    results["year1"] = generate_year_rankings(analysis_1, messages_1, "Year 1")
    
    print("Generating year 2 rankings...")
    results["year2"] = generate_year_rankings(analysis_2, messages_2, "Year 2")
    
    # Comparison rankings between years
    print("Generating year-over-year comparisons...")
    results["comparison"] = generate_comparison_rankings(analysis_1, analysis_2)
    
    # Reaction-based message rankings
    print("Generating reaction-based rankings...")
    results["reactions"] = generate_reaction_rankings(messages_1, messages_2)
    
    # Save all results to files
    save_results(results, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    return results

def generate_year_rankings(analysis, messages, year_label):
    """Generate all rankings for a single year's data"""
    results = {}
    
    # Get user data into DataFrame for easier manipulation
    users_data = []
    for user_id, stats in analysis["users"].items():
        # Skip users with no messages
        if stats["message_count"] == 0:
            continue
            
        # Calculate avg messages per day
        total_days = analysis["metadata"]["total_days"]
        msgs_per_day = stats["message_count"] / total_days if total_days > 0 else 0
        
        # Calculate percent of unliked messages
        unliked_percent = (stats["unliked_messages"] / stats["message_count"]) * 100 if stats["message_count"] > 0 else 0
            
        user_data = {
            'user_id': user_id,
            'nickname': stats['nickname'],
            'real_name': stats['real_name'],
            'message_count': stats['message_count'],
            'total_likes': stats['total_likes'],
            'avg_likes_per_message': stats['avg_likes_per_message'],
            'msgs_per_day': msgs_per_day,
            'unliked_messages': stats['unliked_messages'],
            'unliked_percent': unliked_percent,
            'longest_unliked_streak': stats.get('longest_unliked_streak', 0),
            'longest_liked_streak': stats.get('longest_liked_streak', 0),
            'likes_given': stats.get('likes_given', 0)
        }
        
        users_data.append(user_data)
    
    df = pd.DataFrame(users_data)
    
    # Only include users with at least 5 messages for meaningful rankings
    df_filtered = df[df['message_count'] >= 5].copy()
    
    # 1. Top 10 Likes Per Message Average
    top_lpm = df_filtered.sort_values('avg_likes_per_message', ascending=False).head(10)
    results["top_lpm"] = format_ranking(top_lpm, 'avg_likes_per_message', f"{year_label} Top Likes Per Message")
    
    # 2. Bottom 10 Likes Per Message Average
    bottom_lpm = df_filtered.sort_values('avg_likes_per_message').head(10)
    results["bottom_lpm"] = format_ranking(bottom_lpm, 'avg_likes_per_message', f"{year_label} Lowest Likes Per Message")
    
    # 3. Top 10 Messages Per Day
    top_mpd = df_filtered.sort_values('msgs_per_day', ascending=False).head(10)
    results["top_mpd"] = format_ranking(top_mpd, 'msgs_per_day', f"{year_label} Highest Messages Per Day")
    
    # 4. Top 10 Most Messages Sent
    top_msgs = df_filtered.sort_values('message_count', ascending=False).head(10)
    results["top_messages"] = format_ranking(top_msgs, 'message_count', f"{year_label} Most Messages Sent")
    
    # 5. Top 10 Highest Unliked Message Percentage
    top_unliked = df_filtered.sort_values('unliked_percent', ascending=False).head(10)
    results["top_unliked_percent"] = format_ranking(top_unliked, 'unliked_percent', f"{year_label} Highest Unliked Message %")
    
    # 6. Top 10 Longest Unliked Message Streak
    top_unliked_streak = df_filtered.sort_values('longest_unliked_streak', ascending=False).head(10)
    results["top_unliked_streak"] = format_ranking(top_unliked_streak, 'longest_unliked_streak', f"{year_label} Longest Unliked Message Streak")
    
    # 7. Top CAA (Standardized LPM + Standardized Message Count)
    # Calculate Z-scores for normalization
    if len(df_filtered) > 1:  # Need more than 1 user for meaningful z-scores
        df_filtered['lpm_z'] = (df_filtered['avg_likes_per_message'] - df_filtered['avg_likes_per_message'].mean()) / df_filtered['avg_likes_per_message'].std()
        df_filtered['msg_z'] = (df_filtered['message_count'] - df_filtered['message_count'].mean()) / df_filtered['message_count'].std()
        df_filtered['caa_score'] = df_filtered['lpm_z'] + df_filtered['msg_z']
        
        top_caa = df_filtered.sort_values('caa_score', ascending=False).head(10)
        results["top_caa"] = format_ranking(top_caa, 'caa_score', f"{year_label} Top CAA Score")
    
    # 8. Top 10 Most Generous Likers
    top_likers = df_filtered.sort_values('likes_given', ascending=False).head(10)
    results["top_likers"] = format_ranking(top_likers, 'likes_given', f"{year_label} Most Likes Given")
    
    # 9. Top 10 Least Generous Likers
    bottom_likers = df_filtered.sort_values('likes_given').head(10)
    results["bottom_likers"] = format_ranking(bottom_likers, 'likes_given', f"{year_label} Least Likes Given")
    
    # 10. Most Active Days
    daily_messages = analysis["metadata"]["daily_messages"]
    daily_df = pd.DataFrame(list(daily_messages.items()), columns=['date', 'count'])
    daily_df = daily_df.sort_values('count', ascending=False).head(10)
    
    results["most_active_days"] = {
        "title": f"{year_label} Most Active Days",
        "data": [{
            "rank": i+1,
            "date": date,
            "message_count": count
        } for i, (date, count) in enumerate(zip(daily_df['date'], daily_df['count']))]
    }
    
    return results

def generate_comparison_rankings(analysis_1, analysis_2):
    """Generate rankings comparing metrics between two time periods
    Note: analysis_1 is the earlier year (Year 1), analysis_2 is the later year (Year 2)
    """
    results = {}
    
    # Get user data that exists in both periods
    common_users = {}
    
    for user_id in analysis_1["users"]:
        if user_id in analysis_2["users"]:
            user1 = analysis_1["users"][user_id]  # Year 1 (earlier year)
            user2 = analysis_2["users"][user_id]  # Year 2 (later year)
            
            # Only include users with at least 5 messages in both periods
            if user1["message_count"] >= 5 and user2["message_count"] >= 5:
                # Calculate percent change
                if user1["avg_likes_per_message"] > 0:
                    percent_change = ((user2["avg_likes_per_message"] / user1["avg_likes_per_message"]) - 1) * 100
                else:
                    # Handle division by zero - if previous LPM was 0
                    percent_change = float('inf') if user2["avg_likes_per_message"] > 0 else 0
                
                common_users[user_id] = {
                    "user_id": user_id,
                    "nickname": user2["nickname"],  # Use most recent nickname
                    "real_name": user2["real_name"],
                    "year1_lpm": user1["avg_likes_per_message"],
                    "year2_lpm": user2["avg_likes_per_message"],
                    "lpm_change": user2["avg_likes_per_message"] - user1["avg_likes_per_message"],
                    "lpm_percent_change": percent_change
                }
    
    # Convert to DataFrame
    if common_users:
        df = pd.DataFrame(list(common_users.values()))
        
        # LPM Growers (by percent change) - top 10
        lpm_growers = df.sort_values('lpm_percent_change', ascending=False).head(10)
        results["lpm_growers"] = format_ranking(lpm_growers, 'lpm_percent_change', "Top LPM Growers (% Change)")
        
        # LPM Shrinkers (by percent change) - top 10
        lpm_shrinkers = df.sort_values('lpm_percent_change').head(10)
        results["lpm_shrinkers"] = format_ranking(lpm_shrinkers, 'lpm_percent_change', "Top LPM Shrinkers (% Change)")
    
    return results

def generate_reaction_rankings(messages_1, messages_2):
    """Generate rankings based on message reactions"""
    results = {}
    
    # Combine messages from both periods
    all_messages = messages_1 + messages_2
    
    # Extract messages with user info and reaction data
    processed_messages = []
    
    for msg in all_messages:
        # Get basic message info
        message_info = {
            "id": msg.get("id", ""),
            "user_id": msg.get("user_id", ""),
            "name": msg.get("name", "Unknown User"),
            "text": msg.get("text", ""),
            "created_at": msg.get("created_at", 0),
            "date": datetime.fromtimestamp(msg.get("created_at", 0)).strftime("%Y-%m-%d %H:%M:%S"),
            "likes": len(msg.get("favorited_by", [])),
            "question_marks": 0,
            "thumbs_down": 0,
            "attachments": []
        }
        
        # Add attachment information
        if "attachments" in msg and msg["attachments"]:
            message_info["attachments"] = msg["attachments"]
        
        # Process reactions
        for reaction in msg.get("reactions", []):
            if reaction.get("type") == "emoji":
                # Check for question mark emoji
                if reaction.get("code") == "\u2753":
                    message_info["question_marks"] = len(reaction.get("user_ids", []))
                # Check for thumbs down emoji
                elif reaction.get("code") == "\ud83d\udc4e":
                    message_info["thumbs_down"] = len(reaction.get("user_ids", []))
        
        processed_messages.append(message_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_messages)
    
    # Top 30 most liked messages
    top_liked = df.sort_values('likes', ascending=False).head(30)
    results["top_liked_messages"] = format_message_ranking(top_liked, 'likes', "Top 30 Most Liked Messages")
    
    # Check if we have thumbs_down data
    if df['thumbs_down'].sum() > 0:
        top_thumbs_down = df.sort_values('thumbs_down', ascending=False).head(30)
        results["top_thumbs_down"] = format_message_ranking(top_thumbs_down, 'thumbs_down', "Top 30 Most Thumbs-Down Messages")
    
    # Check if we have question mark data
    if df['question_marks'].sum() > 0:
        top_question_marks = df.sort_values('question_marks', ascending=False).head(30)
        results["top_question_marks"] = format_message_ranking(top_question_marks, 'question_marks', "Top 30 Most Question-Marked Messages")
    
    return results

def format_ranking(df, metric_column, title):
    """Format a DataFrame ranking into a structured dictionary"""
    result = {
        "title": title,
        "data": []
    }
    
    for i, (_, row) in enumerate(df.iterrows()):
        entry = {
            "rank": i+1,
            "nickname": row['nickname'],
            "real_name": row['real_name'],
            "value": row[metric_column]
        }
        result["data"].append(entry)
    
    return result

def format_message_ranking(df, metric_column, title):
    """Format a message DataFrame ranking into a structured dictionary"""
    result = {
        "title": title,
        "data": []
    }
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Truncate very long messages for readability
        text = row['text'] if isinstance(row['text'], str) else ""
        if len(text) > 100:
            text = text[:97] + "..."
            
        entry = {
            "rank": i+1,
            "name": row['name'],
            "text": text,
            "date": row['date'],
            "value": row[metric_column],
            "attachments": []
        }
        
        # Include attachment information
        if 'attachments' in row and row['attachments']:
            entry["attachments"] = row['attachments']
        
        result["data"].append(entry)
    
    return result

def save_results(results, output_dir):
    """Save all ranking results to files"""
    # Save full JSON with all rankings
    with open(f"{output_dir}/all_rankings.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Save individual CSV files for each ranking
    for year_key, year_rankings in results.items():
        year_dir = f"{output_dir}/{year_key}"
        os.makedirs(year_dir, exist_ok=True)
        
        for ranking_key, ranking_data in year_rankings.items():
            # Create DataFrame from the ranking data
            df = pd.DataFrame(ranking_data["data"])
            
            # Save to CSV
            df.to_csv(f"{year_dir}/{ranking_key}.csv", index=False)
            
            # Also save a readable text file
            with open(f"{year_dir}/{ranking_key}.txt", 'w', encoding='utf-8') as f:
                f.write(f"{ranking_data['title']}\n")
                f.write("=" * len(ranking_data['title']) + "\n\n")
                
                for item in ranking_data["data"]:
                    if 'nickname' in item and 'real_name' in item:
                        f.write(f"{item['rank']}. {item['nickname']} ({item['real_name']}): {item['value']}\n")
                    elif 'name' in item and 'text' in item:
                        f.write(f"{item['rank']}. {item['name']} [{item['date']}]: {item['value']}\n")
                        f.write(f"   \"{item['text']}\"\n")
                        
                        # Include media attachments if present
                        if 'attachments' in item and item['attachments']:
                            for attachment in item['attachments']:
                                if attachment.get('type') == 'image' and 'url' in attachment:
                                    f.write(f"   Image: {attachment['url']}\n")
                                elif 'type' in attachment and 'url' in attachment:
                                    f.write(f"   {attachment['type'].capitalize()}: {attachment['url']}\n")
                        f.write("\n")
                    else:
                        # For dates or other formats
                        values = [f"{k}: {v}" for k, v in item.items() if k != 'rank']
                        f.write(f"{item['rank']}. {', '.join(values)}\n")


if __name__ == "__main__":
    # Example usage
    run_analysis(
        "YEAR1.json", 
        "analysis/LAST_YEAR_BASE_ANALYSIS_20250421_113800.json",
        "THIS_YEAR_MESSAGES.json", 
        "analysis/THIS_YEAR_BASE_ANALYSIS_20250421_113947.json",
        "groupme_rankings"
    )