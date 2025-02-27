
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateApplicationBuilder();
builder.Services.AddChatClient(new OllamaChatClient(new Uri("http://localhost:11434"), "deepseek-r1:1.5b"));
var app = builder.Build();

var chatClient = app.Services.GetRequiredService<IChatClient>();
var prompt = new ChatMessage(ChatRole.User,
$$"""
You will be given a text and an output format.
You summarize a given text with a maximum of 100 words, find tags to classify it and assign it to one of the following categories:
- Fictional
- News
- Technical
- Scientific

# Output Format:

Give your answer in the following format in RFC8259 compliant JSON, no deviation from this format is allowed. Do not use more than 3 tags. Do not use other categories than the ones supplied.

{
    "Summary": "The summary of the text",
    "Tags": ["A List", "of", "Tags"],
    "Category": "Fictional|News|Technical|Scientific"   
}

# Input Text:
{{File.ReadAllText("inputFictional.txt")}}

""");

await foreach (var item in chatClient.GetStreamingResponseAsync(prompt))
{
    Console.Write(item.Text);
}