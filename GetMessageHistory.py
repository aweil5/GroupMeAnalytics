from collections import defaultdict
import os
from dotenv import load_dotenv
import httpx
import json
import time
from datetime import datetime, timedelta
from ReviewAnalytics import run_analysis

load_dotenv()
API_KEY = os.getenv("API_KEY")
GROUP_NAME = os.getenv("GROUP_NAME")
BASE_URL = "https://api.groupme.com/v3"
START_DATE = datetime(2024, 8, 15)
END_DATE = datetime.now()
# START_DATE = datetime(2024, 8, 15)
# END_DATE = datetime.now()


def get_group_id(group_name):
    url = f"{BASE_URL}/groups"

    response = httpx.get(url, params={"token": API_KEY})
    response_json = response.json()
    print(response_json)
    for group in response_json["response"]:
        print(group)
        if group["name"] == group_name:
            return group
    return None


def get_member_dict(group_id):
    url = f"{BASE_URL}/groups/{group_id}"

    response = httpx.get(url, params={"token": API_KEY})
    response_json = response.json()
    members = response_json["response"]["members"]
    member_dict = {}
    for member in members:
        member_dict[member["user_id"]] = {
            "nickname": member["nickname"],
            "real_name": member["name"],
        }
    return member_dict


def retrieve_group_messages(group_id, start_date, end_date):
    rate_limit = 2
    url = f"{BASE_URL}/groups/{group_id}/messages"

    all_messages = []
    last_message_id = None
    flip = False

    while True:
        # Prepare parameters for the request
        params = {"token": API_KEY, "limit": 100}  # Use maximum limit for efficiency

        # Add before_id parameter for pagination
        if last_message_id:
            params["before_id"] = last_message_id

        try:
            if flip:
                print("Retrieving messages...")
                flip = False
            else:
                print("Retrieving messages..")
                flip = True

            response = httpx.get(url, params=params)

            # Handle rate limiting
            if response.status_code == 429:
                print(f"Rate limited, waiting {rate_limit} seconds")
                time.sleep(rate_limit)  # Wait 1 minute if rate limited
                rate_limit = rate_limit * 2  # Double wait time for each rate limit
                continue

            # Handle no more messages
            if response.status_code == 304:
                print("No more messages")
                break
            rate_limit = 2  # Reset rate limit
            response.raise_for_status()

            messages = response.json()["response"]["messages"]

            # If no messages returned, we're done
            if not messages:
                break

            # Filter messages by date
            for message in messages:
                message_date = datetime.fromtimestamp(message["created_at"])

                # If we've gone past our start date, we're done
                if message_date < start_date:
                    return all_messages

                # If message is within our date range, add it
                if start_date <= message_date <= end_date:
                    all_messages.append(message)
                else:
                    print(f"Message out of range: {message_date}")

            # Update last_message_id for pagination
            last_message_id = messages[-1]["id"]

        except httpx.HTTPError as e:
            print(f"Error retrieving messages: {e}")
            break

    return all_messages


def get_total_engagement(message):
    """Calculate total engagement for a message (likes + reactions)"""
    # Get likes from favorited_by
    # likes = len(message.get("favorited_by", []))

    # Get reactions count
    reaction_count = sum(
        len(reaction["user_ids"]) for reaction in message.get("reactions", [])
    )

    return reaction_count

    # analytics_results = scrape_analytics(
    #     group_id,
    #     START_DATE,
    #     end_date=END_DATE,
    #     use_cache=True,
    #     cache_filename="YEAR1.json",
    # )
def scrape_analytics(
    group_id,
    start_date,
    end_date=None,
    use_cache=False,
    cache_filename="messages.json",
):
    member_dict = get_member_dict(group_id)
    
    if not use_cache:
        messages = retrieve_group_messages(group_id, datetime(1970, 1, 1), datetime.now())
        with open(cache_filename, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, default=str)
    else:
        with open(cache_filename, "r") as f:
            messages = json.load(f)
    
    # Determine the actual date range from the messages
    if not messages:
        print("No messages found in the dataset")
        return None
    
    # Sort messages by created_at timestamp
    messages.sort(key=lambda m: m["created_at"])
    
    # Get the actual start and end dates from the messages
    start_date = datetime.fromtimestamp(messages[0]["created_at"])
    end_date = datetime.fromtimestamp(messages[-1]["created_at"])
    
    print(f"Actual date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    analytics = {
        "metadata": {
            "total_days": (end_date - start_date).days + 1,  # +1 to include both start and end dates
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "daily_messages": {},
        },
        "users": {},
    }

    for message in messages:
        message_date = datetime.fromtimestamp(message["created_at"])
        date_str = message_date.strftime("%Y-%m-%d")
        if date_str not in analytics["metadata"]["daily_messages"]:
            analytics["metadata"]["daily_messages"][date_str] = 0
        analytics["metadata"]["daily_messages"][date_str] += 1

        user_id = message["user_id"]

        if (
            user_id not in member_dict
            or member_dict[user_id]["nickname"] == member_dict[user_id]["real_name"]
        ):
            member_dict[user_id] = {
                "nickname": "FORMER_MEMBER",
                "real_name": message.get("name", "Former Member"),
            }
            print(f"Adding user to the member dictionary: {member_dict[user_id]}")

        if user_id not in analytics["users"] and user_id in member_dict:
            analytics["users"][user_id] = {
                "nickname": member_dict[user_id]["nickname"],
                "real_name": member_dict[user_id]["real_name"],
                "message_count": 0,
                "last_message": None,
                "total_likes": 0,
                "most_liked_message": None,
                "most_liked_message_likes": 0,
                "avg_likes_per_message": 0,
                "total_words": 0,
                "avg_words_per_message": 0,
                "messages_by_hour": {str(i).zfill(2): 0 for i in range(24)},
                "messages_by_day": {
                    "Monday": 0,
                    "Tuesday": 0,
                    "Wednesday": 0,
                    "Thursday": 0,
                    "Friday": 0,
                    "Saturday": 0,
                    "Sunday": 0,
                },
                "likes_given": 0,
                "unliked_messages": 0,
                "current_unliked_streak": 0,
                "longest_unliked_streak": 0,
                "current_liked_streak": 0,
                "longest_liked_streak": 0,
                "daily_messages": {},
                "top_liked_content": [],
            }

        user_stats = analytics["users"][user_id]
        user_stats["message_count"] += 1

        if date_str not in user_stats["daily_messages"]:
            user_stats["daily_messages"][date_str] = 0
        user_stats["daily_messages"][date_str] += 1

        likes_count = len(message.get("favorited_by", []))

        # Track like/unlike streaks
        if likes_count == 0:
            user_stats["unliked_messages"] += 1
            user_stats["current_unliked_streak"] += 1
            user_stats["current_liked_streak"] = 0

            if (
                user_stats["current_unliked_streak"]
                > user_stats["longest_unliked_streak"]
            ):
                user_stats["longest_unliked_streak"] = user_stats[
                    "current_unliked_streak"
                ]
        else:
            user_stats["current_unliked_streak"] = 0
            user_stats["current_liked_streak"] += 1

            if (
                user_stats["current_liked_streak"]
                > user_stats["longest_liked_streak"]
            ):
                user_stats["longest_liked_streak"] = user_stats[
                    "current_liked_streak"
                ]

            # Track top liked content
            if "top_liked_content" not in user_stats:
                user_stats["top_liked_content"] = []

            if len(user_stats["top_liked_content"]) < 5:
                user_stats["top_liked_content"].append(message)
            else:
                least_liked = 500
                index_to_replace = -1
                for i in range(len(user_stats["top_liked_content"])):
                    curr_likes = get_total_engagement(
                        user_stats["top_liked_content"][i]
                    )
                    if curr_likes < least_liked:
                        least_liked = curr_likes
                        index_to_replace = i
                if index_to_replace != -1 and likes_count > least_liked:
                    user_stats["top_liked_content"][index_to_replace] = message

        # Update last message
        if user_stats[
            "last_message"
        ] is None or message_date > datetime.fromtimestamp(
            user_stats["last_message"]["created_at"]
        ):
            user_stats["last_message"] = message

        # Update likes
        user_stats["total_likes"] += likes_count

        # Update most liked message
        if likes_count > user_stats["most_liked_message_likes"]:
            user_stats["most_liked_message"] = message
            user_stats["most_liked_message_likes"] = likes_count

        # Update word count
        if message.get("text"):
            words = len(message["text"].split())
            user_stats["total_words"] += words

        # Update time-based analytics
        hour = message_date.strftime("%H")
        day = message_date.strftime("%A")
        user_stats["messages_by_hour"][hour] += 1
        user_stats["messages_by_day"][day] += 1

        # Track likes given
        for liker_id in message.get("favorited_by", []):
            if liker_id in analytics["users"]:
                analytics["users"][liker_id]["likes_given"] += 1

    # Calculate averages
    for user_id, stats in analytics["users"].items():
        if stats["message_count"] > 0:
            stats["avg_likes_per_message"] = (
                stats["total_likes"] / stats["message_count"]
            )
            stats["avg_words_per_message"] = (
                stats["total_words"] / stats["message_count"]
            )
        else:
            print(f"Warning: No messages found for {stats['nickname']}")

    # Remove all former members from the analytics
    analytics["users"] = {
        user_id: stats
        for user_id, stats in analytics["users"].items()
        if stats["nickname"] != "FORMER_MEMBER"
    }

    return analytics


# def get_training_data(group_id, start_date, end_date):
#     print(f"Obraining Data over {start_date} to {end_date}")
#     print(f"This will be {end_date - start_date} days of data")

#     messages = retrieve_group_messages(group_id, start_date, end_date)
#     with open("training.json", "w", encoding="utf-8") as f:
#         json.dump(messages, f, indent=2, default=str)


# Example usage:
if __name__ == "__main__":
    main_group_info = get_group_id(GROUP_NAME)
    if not main_group_info:
        print(f"Could not find group: {GROUP_NAME}")
        exit(1)

    group_id = main_group_info["id"]

    analytics_results = scrape_analytics(
        group_id,
        START_DATE,
        end_date=END_DATE,
        use_cache=True,
        cache_filename="THIS_YEAR_MESSAGES.json",
    )

    # Save results to a JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"THIS_YEAR_BASE_ANALYSIS_{timestamp}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(analytics_results, f, indent=2, default=str)

    print(f"Analytics saved to {output_filename}")

    # print("Analyzing Data")
    # run_analysis(output_filename, True)
