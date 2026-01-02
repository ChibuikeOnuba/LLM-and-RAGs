result = model.invoke(messages)

# Access usage information
if hasattr(result, 'usage_metadata'):
    prompt_tokens = result.usage_metadata.get('prompt_tokens', 0)
    completion_tokens = result.usage_metadata.get('completion_tokens', 0)
    total_tokens = result.usage_metadata.get('total_tokens', 0)
    
    # Calculate cost (GPT-4 pricing example)
    input_cost = prompt_tokens * 0.00003  # $0.03 per 1K tokens
    output_cost = completion_tokens * 0.00006  # $0.06 per 1K tokens
    total_cost = input_cost + output_cost
    
    print(f"Cost for this query: ${total_cost:.4f}")
    print(f"Tokens used: {total_tokens}")

2. Monitor Response Quality & Performance
pythonimport time

start_time = time.time()
result = model.invoke(messages)
response_time = time.time() - start_time

# Log performance metrics
metrics = {
    'query': query,
    'response_time': response_time,
    'tokens_used': result.usage_metadata.get('total_tokens', 0),
    'model': result.response_metadata.get('model_name', 'unknown'),
    'finish_reason': result.response_metadata.get('finish_reason', 'unknown')
}

# Save to database or log file
print(f"Response generated in {response_time:.2f}s")

3. Detect Truncated Responses
pythonresult = model.invoke(messages)

# Check if response was cut off due to token limit
finish_reason = result.response_metadata.get('finish_reason', '')

if finish_reason == 'length':
    print("⚠️ Warning: Response was truncated due to token limit!")
    print("Consider: Reducing document size or increasing max_tokens")
elif finish_reason == 'stop':
    print("✅ Response completed normally")

4. Build Usage Analytics Dashboard
python# Store metadata for each query
query_log = []

result = model.invoke(messages)

query_log.append({
    'timestamp': datetime.now(),
    'user_id': user_id,
    'query_length': len(query),
    'tokens_used': result.usage_metadata.get('total_tokens', 0),
    'cost': calculate_cost(result),
    'response_time': response_time,
    'model': result.response_metadata.get('model_name')
})

# Later: Analyze patterns
# - Which users consume most tokens?
# - What time of day has highest usage?
# - Average cost per query
# - Identify expensive queries to optimize

5. Implement Budget Controls
pythonclass RAGWithBudget:
    def __init__(self, daily_budget=10.0):
        self.daily_budget = daily_budget
        self.daily_spent = 0.0
    
    def query(self, messages):
        if self.daily_spent >= self.daily_budget:
            raise Exception(f"Daily budget of ${self.daily_budget} exceeded!")
        
        result = model.invoke(messages)
        
        # Calculate and track cost
        tokens = result.usage_metadata.get('total_tokens', 0)
        cost = tokens * 0.00004  # Average cost
        self.daily_spent += cost
        
        print(f"Remaining budget: ${self.daily_budget - self.daily_spent:.2f}")
        return result

6. Optimize Document Selection
python# Track which queries use most tokens
result = model.invoke(messages)
tokens_used = result.usage_metadata.get('prompt_tokens', 0)

if tokens_used > 8000:  # High token usage
    print(f"⚠️ Query used {tokens_used} tokens")
    print("Consider: Retrieving fewer documents or shorter chunks")
    
    # Automatically reduce documents for next similar query
    max_docs = 3  # Instead of 5

7. A/B Testing Different Models
pythondef compare_models(query, relevant_docs):
    models_to_test = ['gpt-4o', 'gpt-3.5-turbo']
    results = {}
    
    for model_name in models_to_test:
        model = ChatOpenAI(model=model_name)
        result = model.invoke(messages)
        
        results[model_name] = {
            'answer': result.content,
            'cost': calculate_cost(result),
            'tokens': result.usage_metadata.get('total_tokens', 0),
            'quality_score': evaluate_answer(result.content)
        }
    
    # Choose best model based on cost/quality tradeoff
    return results

8. Debug & Logging
pythonimport logging

result = model.invoke(messages)

# Comprehensive logging
logging.info(f"""
Query processed:
- Model: {result.response_metadata.get('model_name')}
- Tokens: {result.usage_metadata.get('total_tokens', 0)}
- Finish reason: {result.response_metadata.get('finish_reason')}
- Response ID: {result.id}
- Documents used: {len(relevant_docs)}
""")

# Helps troubleshoot issues later

9. Rate Limiting & Throttling
pythonclass RateLimitedRAG:
    def __init__(self, max_tokens_per_minute=90000):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens_used_this_minute = 0
        self.minute_start = time.time()
    
    def query(self, messages):
        # Reset counter every minute
        if time.time() - self.minute_start > 60:
            self.tokens_used_this_minute = 0
            self.minute_start = time.time()
        
        result = model.invoke(messages)
        tokens = result.usage_metadata.get('total_tokens', 0)
        
        self.tokens_used_this_minute += tokens
        
        if self.tokens_used_this_minute > self.max_tokens_per_minute:
            print("⚠️ Approaching rate limit, slowing down...")
            time.sleep(60)
        
        return result

Most Practical Use Cases

Cost tracking - Know how much each query costs
Performance monitoring - Identify slow queries
Budget management - Prevent overspending
Quality assurance - Detect truncated responses
Optimization - Find queries that use too many tokens


Quick Example: Production-Ready Logging
pythondef query_with_logging(query, relevant_docs):
    start_time = time.time()
    
    result = model.invoke(messages)
    
    # Extract metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'query': query[:100],  # First 100 chars
        'response_time': time.time() - start_time,
        'tokens': result.usage_metadata.get('total_tokens', 0),
        'cost': calculate_cost(result),
        'finish_reason': result.response_metadata.get('finish_reason'),
        'model': result.response_metadata.get('model_name')
    }
    
    # Save to database/file
    save_to_analytics_db(metadata)
    
    return result.content