import autogen

llm_config = {"model": "gpt-3.5-turbo"}

task = '''
        Write a concise but engaging summary of the most relevant
        news and updates based on timeliness and the users suggested interests.
       '''

news_anchor = autogen.AssistantAgent(
    name="News Anchor",
    system_message="You are a news anchor. You write engaging and accurate " 
        "news on the most relevant topics of the day. You must incorporate "
        "stories from experts on each topic and prioritize only the most important "
        "or relevant stories told in a clear and consise way.",
    llm_config=llm_config,
)

reply = news_anchor.generate_reply(messages=[{"content": task, "role": "user"}])

editor = autogen.AssistantAgent(
    name="Editor",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content.",
)

Sports_reporter = autogen.AssistantAgent(
    name="Sports reporter",
    llm_config=llm_config,
    system_message="You are a sports reporter, you are known for "
        "your ability to comment on the most interesting and "
        "relevant news for recent games, upcoming matches and sports news. " 
        "Suggest 3 of the most interesting and relevant stories for the day, "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

News_Reporter = autogen.AssistantAgent(
    name="News Reporter",
    llm_config=llm_config,
    system_message="You are a news reporter, you are known for "
        "your ability to find the most important stories "
        "from world and local events, politics, science and other hard news topics. " 
        "Suggest 3 of the most interesting and relevant stories for the day, "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

Finance_Reporter = autogen.AssistantAgent(
    name="Finance Reporter",
    llm_config=llm_config,
    system_message="You are a finance reporter, you are known for "
        "your ability to find the most important stories "
        "from world and local events, politics, science and other hard news topics. " 
        "Suggest 3 of the most interesting and relevant stories for the day, "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

meta_reviewer = autogen.AssistantAgent(
    name="Meta Reviewer",
    llm_config=llm_config,
    system_message="You are a meta reviewer, you aggragate and review "
    "the work of other reviewers and give a final suggestion on the content.",
)

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

review_chats = [
    {
     "recipient": Sports_reporter, 
     "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into as JSON object only:"
        "{'Stories': [{ 'Story': '' }]}. Here Reporter should be your role",},
     "max_turns": 1},
    {
    "recipient": News_Reporter, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return stories into as JSON object only:"
        "{'Stories': [{ 'Story': '' }]}",},
     "max_turns": 1},
    {"recipient": Finance_Reporter, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return stories into as JSON object only:"
        "{'Stories': [{ 'Story': '' }]}",},
     "max_turns": 1},
     {"recipient": meta_reviewer, 
      "message": "Aggregrate stories from all reporters and give final suggestions of stories to report.", 
     "max_turns": 1},
]

editor.register_nested_chats(
    review_chats,
    trigger=news_anchor,
)

res = editor.initiate_chat(
    recipient=news_anchor,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)

print(res.summary)
