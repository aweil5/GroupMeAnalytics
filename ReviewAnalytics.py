from datetime import datetime
import json
import numpy as np
# from textblob import TextBlob
import re
from collections import defaultdict


class GroupMeAnalytics:
    def __init__(self, analytics_data):
        self.analytics_data = analytics_data
        self.group_context = None
        self.enhanced_data = None
        
    def process_analytics(self):
        """Main processing function that coordinates all analytics"""
        # First calculate group context for normalization
        messages = self._collect_all_messages()
        self.group_context = calculate_group_context(messages, self.analytics_data["users"])
        
        # Then enhance the data with improved CAA scores
        self.enhanced_data = self._enhance_analytics()
        return self.enhanced_data
        
    def _collect_all_messages(self):
        """Collect all messages from user data"""
        messages = []
        for user_id, user_data in self.analytics_data["users"].items():
            # Add complete message objects
            if user_data.get("last_message"):
                messages.append(user_data["last_message"])
            if user_data.get("most_liked_message"):
                messages.append(user_data["most_liked_message"])
                
            # Handle unliked messages more carefully
            for msg in user_data.get("unliked_message_content", []):
                if isinstance(msg, dict) and "text" in msg:
                    # Create a more complete message object
                    full_msg = {
                        "text": msg["text"],
                        "created_at": msg.get("created_at", 0),
                        "favorited_by": [],  # These are unliked messages
                        "replies": msg.get("replies", []),
                        "attachments": msg.get("attachments", []),
                        "id": msg.get("id", ""),
                        "user_id": user_id
                    }
                    messages.append(full_msg)
        
        return messages
        
    def _enhance_analytics(self):
        """Enhance analytics with improved CAA metrics"""
        enhanced_data = self.analytics_data.copy()
        
        # Process each user's data
        for user_id, user_stats in enhanced_data["users"].items():
            if user_stats["message_count"] > 0:
                # Calculate improved CAA score for each message
                message_scores = []
                
                # Score last message if available
                if user_stats.get("last_message"):
                    last_msg_score = calculate_improved_caa_score(
                        user_stats["last_message"], 
                        self.group_context
                    )
                    message_scores.append(last_msg_score)
                
                # Score most liked message if available
                if user_stats.get("most_liked_message"):
                    liked_msg_score = calculate_improved_caa_score(
                        user_stats["most_liked_message"], 
                        self.group_context
                    )
                    message_scores.append(liked_msg_score)
                
                # Score unliked messages
                for msg in user_stats.get("unliked_message_content", []):
                    if isinstance(msg, dict) and "text" in msg:
                        unliked_score = calculate_improved_caa_score(msg, self.group_context)
                        message_scores.append(unliked_score)
                
                # Calculate average CAA score
                if message_scores:
                    avg_caa_score = np.mean(message_scores)
                    consistency_score = 1 / (1 + np.std(message_scores)) if len(message_scores) > 1 else 1
                else:
                    avg_caa_score = 0
                    consistency_score = 0
                
                # Update user stats with new metrics
                user_stats.update({
                    "caa_score": round(avg_caa_score, 2),
                    "activity_consistency": round(consistency_score, 2),
                    "avg_daily_score": round(np.mean(message_scores) if message_scores else 0, 2)
                })
        
        # Calculate group-wide statistics
        all_caa_scores = [user["caa_score"] for user in enhanced_data["users"].values()]
        if all_caa_scores:
            caa_threshold = np.mean(all_caa_scores) + np.std(all_caa_scores)
            
            # Update CAA status for each user
            for user_stats in enhanced_data["users"].values():
                user_stats["is_caa"] = user_stats["caa_score"] > caa_threshold
            
            # Update global statistics
            enhanced_data["metadata"].update({
                "avg_group_caa": np.mean(all_caa_scores),
                "caa_threshold": caa_threshold,
                "top_contributors": get_top_contributors(enhanced_data["users"])
            })
        
        return enhanced_data

def calculate_improved_caa_score(message, group_context):
    """Calculate a more nuanced Content Above Average score without TextBlob"""
    base_score = 0
    text = message.get("text", "")
    if not text:
        return 0
        
    # 1. Engagement Metrics (40% of score)
    likes = len(message.get("favorited_by", []))
    likes_score = min(likes / (group_context["avg_likes"] or 1) * 20, 20)
    replies = len(message.get("replies", []))
    replies_score = min(replies / (group_context["avg_replies"] or 1) * 20, 20)
    
    # 2. Content Quality Metrics (30% of score)
    # Word count normalized against group average
    word_count = len(text.split())
    length_score = min(word_count / (group_context["avg_words"] or 1) * 10, 10)
    
    # Vocabulary diversity
    unique_words = len(set(text.lower().split()))
    uniqueness_score = (unique_words / word_count * 10) if word_count > 0 else 0
    
    # Simple sentence structure (based on punctuation)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    if sentence_count == 0 and len(text) > 0:
        sentence_count = 1
    structure_score = min(sentence_count / (group_context["avg_sentences"] or 1) * 10, 10)
    
    # 3. Media and Link Contribution (15% of score)
    media_score = calculate_media_score(message)
    
    # 4. Timing and Context (15% of score)
    time_score = calculate_timing_score(message, group_context)
    
    # Calculate final weighted score
    total_score = (
        (likes_score + replies_score) * 0.4 +
        (length_score + uniqueness_score + structure_score) * 0.3 +
        media_score * 0.15 +
        time_score * 0.15
    )
    
    return round(total_score, 2)

def calculate_group_context(messages, users_data):
    """Calculate group-wide metrics for context without TextBlob"""
    context = {
        "avg_likes": 0,
        "avg_replies": 0,
        "avg_words": 0,
        "avg_sentences": 0,
        "peak_hours": set(),
        "user_stats": defaultdict(dict)
    }
    
    # Filter out None and invalid messages
    valid_messages = [msg for msg in messages if msg and isinstance(msg, dict)]
    
    # Calculate averages
    total_messages = len(valid_messages)
    if total_messages == 0:
        return context
        
    total_likes = sum(len(msg.get("favorited_by", [])) for msg in valid_messages)
    total_replies = sum(len(msg.get("replies", [])) for msg in valid_messages)
    
    # Handle text processing safely
    total_words = 0
    total_sentences = 0
    for msg in valid_messages:
        text = msg.get("text", "")
        if text and isinstance(text, str):
            total_words += len(text.split())
            # Count sentences based on punctuation
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            if sentence_count == 0 and len(text) > 0:
                sentence_count = 1
            total_sentences += sentence_count
    
    context["avg_likes"] = total_likes / total_messages if total_messages > 0 else 0
    context["avg_replies"] = total_replies / total_messages if total_messages > 0 else 0
    context["avg_words"] = total_words / total_messages if total_messages > 0 else 0
    context["avg_sentences"] = total_sentences / total_messages if total_messages > 0 else 0
    
    # Calculate peak hours
    hour_counts = defaultdict(int)
    for msg in valid_messages:
        if "created_at" in msg:
            try:
                hour = datetime.fromtimestamp(msg["created_at"]).hour
                hour_counts[hour] += 1
            except:
                continue
    
    if hour_counts:
        avg_msgs_per_hour = sum(hour_counts.values()) / 24
        context["peak_hours"] = {
            hour for hour, count in hour_counts.items()
            if count > avg_msgs_per_hour
        }
    
    # Calculate user statistics
    for user_id, stats in users_data.items():
        if stats and isinstance(stats, dict):
            context["user_stats"][user_id] = {
                "daily_messages": stats.get("daily_messages", {}),
                "total_messages": stats.get("message_count", 0)
            }
    
    return context

def calculate_media_score(message):
    """Calculate media contribution score"""
    media_score = 0
    attachments = message.get("attachments", [])
    for attachment in attachments:
        if attachment.get("type") == "image":
            media_score += 5
        elif attachment.get("type") == "video":
            media_score += 7
        elif attachment.get("type") == "link":
            media_score += 3
    return min(media_score, 15)

def calculate_timing_score(message, group_context):
    """Calculate timing and context score"""
    time_score = 0
    hour = datetime.fromtimestamp(message["created_at"]).hour
    
    # Bonus for active hours
    if hour in group_context["peak_hours"]:
        time_score += 5
    
    # Bonus for conversation starters
    if is_conversation_starter(message, group_context):
        time_score += 5
    
    # Bonus for consistent contributors
    if is_consistent_contributor(message.get("user_id"), group_context):
        time_score += 5
        
    return time_score

def is_conversation_starter(message, group_context):
    """Check if message started a significant conversation thread"""
    reply_threshold = group_context["avg_replies"] * 1.5
    return len(message.get("replies", [])) >= reply_threshold

def is_consistent_contributor(user_id, group_context):
    """Check if user is a consistent contributor"""
    user_stats = group_context["user_stats"].get(user_id, {})
    daily_msgs = user_stats.get("daily_messages", {})
    if not daily_msgs:
        return False
        
    # Calculate consistency using coefficient of variation
    msgs_per_day = list(daily_msgs.values())
    if not msgs_per_day:
        return False
        
    cv = np.std(msgs_per_day) / np.mean(msgs_per_day)
    return cv < 1.0  # Lower CV indicates more consistency


def get_top_contributors(users_data, top_n=5):
    """Get the top contributors based on CAA score"""
    return sorted(
        [
            {"user_id": uid, "nickname": data["nickname"], "caa_score": data["caa_score"]}
            for uid, data in users_data.items()
        ],
        key=lambda x: x["caa_score"],
        reverse=True
    )[:top_n]

def print_enhanced_analytics(analytics_data):
    """Print enhanced analytics including CAA metrics and traditional stats"""
    print("\n=== Enhanced GroupMe Analytics ===\n")
    
    # CAA Overview
    print("Content Above Average (CAA) Overview:")
    print(f"Group Average CAA Score: {analytics_data['metadata']['avg_group_caa']:.2f}")
    print(f"CAA Threshold: {analytics_data['metadata']['caa_threshold']:.2f}")
    
    print("\nTop Contributors:")
    for i, contributor in enumerate(analytics_data['metadata']['top_contributors'], 1):
        print(f"{i}. {contributor['nickname']}: {contributor['caa_score']:.2f}")
    
    # Top 5 Most Liked Messages
    print("\nTop 5 Most Liked Messages:")
    liked_messages = []
    for user_id, stats in analytics_data["users"].items():
        if stats.get("most_liked_message") and stats.get("most_liked_message_likes", 0) > 0:
            liked_messages.append({
                "nickname": stats["nickname"],
                "real_name": stats["real_name"],
                "likes": stats["most_liked_message_likes"],
                "message": stats["most_liked_message"].get("text", "[No text]"),
                "created_at": datetime.fromtimestamp(stats["most_liked_message"]["created_at"])
            })
    
    for i, msg in enumerate(sorted(liked_messages, key=lambda x: x["likes"], reverse=True)[:5], 1):
        print(f"\n{i}. From: {msg['nickname']} ({msg['real_name']})")
        print(f"   Likes: {msg['likes']}")
        print(f"   Posted: {msg['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Message: {msg['message']}")
    
    # Message Activity Stats
    print("\nMessage Activity Statistics:")
    
    # Messages per day
    total_days = analytics_data["metadata"]["total_days"]
    user_msg_per_day = []
    for user_id, stats in analytics_data["users"].items():
        if stats["message_count"] > 0:
            msg_per_day = stats["message_count"] / total_days
            user_msg_per_day.append({
                "nickname": stats["nickname"],
                "real_name": stats["real_name"],
                "msg_per_day": msg_per_day,
                "total_messages": stats["message_count"]
            })
    
    print("\nTop 10 Most Active Users (Messages per Day):")
    for i, user in enumerate(sorted(user_msg_per_day, key=lambda x: x["msg_per_day"], reverse=True)[:10], 1):
        print(f"{i}. {user['nickname']} ({user['real_name']}): {user['msg_per_day']:.2f} msgs/day ({user['total_messages']} total)")
    
    # Likes per Message Analysis
    print("\nLikes per Message Analysis:")
    user_likes_per_msg = []
    for user_id, stats in analytics_data["users"].items():
        if stats["message_count"] >= 5:  # Minimum message threshold
            likes_per_msg = stats["total_likes"] / stats["message_count"]
            user_likes_per_msg.append({
                "nickname": stats["nickname"],
                "real_name": stats["real_name"],
                "likes_per_msg": likes_per_msg,
                "total_messages": stats["message_count"]
            })
    
    print("\nTop 10 Most Liked Users (per message, min 5 messages):")
    for i, user in enumerate(sorted(user_likes_per_msg, key=lambda x: x["likes_per_msg"], reverse=True)[:10], 1):
        print(f"{i}. {user['nickname']} ({user['real_name']}): {user['likes_per_msg']:.2f} likes/msg ({user['total_messages']} msgs)")
    
    print("\nBottom 10 Least Liked Users (per message, min 5 messages):")
    for i, user in enumerate(sorted(user_likes_per_msg, key=lambda x: x["likes_per_msg"])[:10], 1):
        print(f"{i}. {user['nickname']} ({user['real_name']}): {user['likes_per_msg']:.2f} likes/msg ({user['total_messages']} msgs)")
    
    # Unliked Messages Analysis
    print("\nUnliked Messages Analysis:")
    user_unliked = []
    for user_id, stats in analytics_data["users"].items():
        if stats["message_count"] >= 5:
            unliked_ratio = stats["unliked_messages"] / stats["message_count"]
            user_unliked.append({
                "nickname": stats["nickname"],
                "real_name": stats["real_name"],
                "unliked_ratio": unliked_ratio,
                "unliked_count": stats["unliked_messages"],
                "total_messages": stats["message_count"],
                "longest_streak": stats["longest_unliked_streak"]
            })
    
    print("\nTop 10 Highest Unliked Message Ratios (min 5 messages):")
    for i, user in enumerate(sorted(user_unliked, key=lambda x: x["unliked_ratio"], reverse=True)[:10], 1):
        print(f"{i}. {user['nickname']} ({user['real_name']}): {user['unliked_ratio']:.2%} ({user['unliked_count']}/{user['total_messages']})")
    
    print("\nTop 10 Longest Unliked Message Streaks:")
    for i, user in enumerate(sorted(user_unliked, key=lambda x: x["longest_streak"], reverse=True)[:10], 1):
        print(f"{i}. {user['nickname']} ({user['real_name']}): {user['longest_streak']} messages")
    
    # Most Active Days
    print("\nTop 10 Most Active Days:")
    daily_messages = analytics_data["metadata"]["daily_messages"]
    sorted_days = sorted(daily_messages.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (date, count) in enumerate(sorted_days, 1):
        print(f"{i}. {date}: {count} messages")
    
    print("\nTop 10 Content Above Average (CAA) Contributors:")
    
    # Collect user CAA data
    caa_users = []
    for user_id, stats in analytics_data["users"].items():
        if stats["message_count"] > 0:
            caa_users.append({
                "nickname": stats["nickname"],
                "real_name": stats["real_name"],
                "caa_score": stats["caa_score"],
                "consistency": stats["activity_consistency"],
                "daily_score": stats["avg_daily_score"]
            })
    
    # Sort by CAA score and get top 10
    top_users = sorted(caa_users, key=lambda x: x["caa_score"], reverse=True)[:10]
    
    # Print results
    for i, user in enumerate(top_users, 1):
        print(f"\n{i}. {user['nickname']} ({user['real_name']}):")
        print(f"   CAA Score: {user['caa_score']:.2f}")
        print(f"   Consistency Score: {user['consistency']:.2f}")
        print(f"   Average Daily Score: {user['daily_score']:.2f}")

def run_analysis(filename):
    """Main function to run the enhanced analytics"""
    try:
        # Load data
        with open(filename, "r", encoding="utf-8") as f:
            analytics_data = json.load(f)
        
        # Process analytics
        analyzer = GroupMeAnalytics(analytics_data)
        enhanced_data = analyzer.process_analytics()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"enhanced_groupme_analytics_{timestamp}.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(enhanced_data, f, indent=2, default=str)
        
        # Print results
        print_enhanced_analytics(enhanced_data)
        
        return enhanced_data
        
    except Exception as e:
        print(f"Error processing analytics: {str(e)}")
        raise

if __name__ == "__main__":
    run_analysis("groupme_analytics_20250209_231637.json")
    # import sys
    # if len(sys.argv) > 1:
    #     run_analysis(sys.argv[1])
    # else:
    #     # Default test file if no argument provided
    #     print("No file provided, attempting to use most recent analytics file...")
    #     try:
    #         # Try to find the most recent analytics file in the current directory
    #         import glob
    #         files = glob.glob("groupme_analytics_*.json")
    #         if files:
    #             latest_file = max(files, key=os.path.getctime)
    #             print(f"Found file: {latest_file}")
    #             run_analysis(latest_file)
    #         else:
    #             print("No analytics files found. Please provide a JSON file path as an argument")
    #     except Exception as e:
    #         print(f"Error processing analytics: {str(e)}")
    #         raise
            