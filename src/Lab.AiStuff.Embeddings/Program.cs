using System.Globalization;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OllamaEmbeddingGenerator(new Uri("http://localhost:11434/"), "nomic-embed-text");

var koenig = await generator.GenerateEmbeddingVectorAsync("König");
var mann = await generator.GenerateEmbeddingVectorAsync("Mann");
var frau = await generator.GenerateEmbeddingVectorAsync("Frau");
var koenigin = await generator.GenerateEmbeddingVectorAsync("Königin");

var koenigMinusMannPlusFrau = AddVectors(SubtractVectors(koenig, mann), frau);
var distance = CosineSimilairty(koenigMinusMannPlusFrau, koenigin);

Console.WriteLine($"Similarity between König - Mann + Frau and Königin: {distance}");
Console.WriteLine("Similarity between König and König: " + CosineSimilairty(koenig, koenig));
Console.WriteLine("Similarity between König and Mann: " + CosineSimilairty(koenig, mann));
Console.WriteLine("Similarity between König and Frau: " + CosineSimilairty(koenig, frau));
Console.WriteLine("Similarity between Mann and Frau: " + CosineSimilairty(mann, frau));
Console.WriteLine("Similarity between König and Königin: " + CosineSimilairty(koenig, koenigin));
Console.WriteLine("Similarity between Mann and Königin: " + CosineSimilairty(mann, koenigin));
Console.WriteLine("Similarity between König-Mann and Königin: " + CosineSimilairty(SubtractVectors(koenig, mann), koenigin));

var recipeData = GetRecipeData();

var vectorStore = new InMemoryVectorStore();

var recipes = vectorStore.GetCollection<int, Recipe>("recipes");

await recipes.CreateCollectionIfNotExistsAsync();

var index = 0;
foreach (var recipe in recipeData.Take(500))
{
    Console.WriteLine(index++);
    recipe.Vector = await generator.GenerateEmbeddingVectorAsync(recipe.Ingredients);
    await recipes.UpsertAsync(recipe);
}

var query = "beef";
var queryEmbedding = await generator.GenerateEmbeddingVectorAsync(query);

var searchOptions = new VectorSearchOptions()
{
    Top = 4,
    VectorPropertyName = "Vector"
};

var results = await recipes.VectorizedSearchAsync(queryEmbedding, searchOptions);

await PrintResult(results);

var resultList = results.Results.ToBlockingEnumerable().ToList();
var result1 = resultList[1];

var withoutBeef = await SubstractTermFromVectorAsync(generator, result1.Record.Vector, "Beef");
var withoutBeefresults = await recipes.VectorizedSearchAsync(withoutBeef, searchOptions);

await PrintResult(withoutBeefresults);

Console.ReadLine();

static async Task PrintResult(VectorSearchResults<Recipe> results)
{
    await foreach (var result in results.Results)
    {
        Console.WriteLine($"Title: {result.Record.Title}");
        Console.WriteLine($"Score: {result.Score}");
        Console.WriteLine();
    }

    Console.WriteLine("-------------------");
}

static async Task<ReadOnlyMemory<float>> SubstractTermFromVectorAsync(IEmbeddingGenerator<string, Embedding<float>> generator, ReadOnlyMemory<float> vector, string term)
{
    var termVector = await generator.GenerateEmbeddingVectorAsync(term);
    var result = new float[vector.Length];
    for (var i = 0; i < vector.Length; i++)
    {
        result[i] = vector.Span[i] - termVector.Span[i];
    }
    return new ReadOnlyMemory<float>(result);
}

static async Task<ReadOnlyMemory<float>> AddTermToVector(IEmbeddingGenerator<string, Embedding<float>> generator, ReadOnlyMemory<float> vector, string term)
{
    var termVector = await generator.GenerateEmbeddingVectorAsync(term);
    var result = new float[vector.Length];
    for (var i = 0; i < vector.Length; i++)
    {
        result[i] = vector.Span[i] + termVector.Span[i];
    }
    return new ReadOnlyMemory<float>(result);
}

static ReadOnlyMemory<float> SubtractVectors(ReadOnlyMemory<float> vector1, ReadOnlyMemory<float> vector2)
{
    var result = new float[vector1.Length];
    for (var i = 0; i < vector1.Length; i++)
    {
        result[i] = vector1.Span[i] - vector2.Span[i];
    }
    return new ReadOnlyMemory<float>(result);
}

static ReadOnlyMemory<float> AddVectors(ReadOnlyMemory<float> vector1, ReadOnlyMemory<float> vector2)
{
    var result = new float[vector1.Length];
    for (var i = 0; i < vector1.Length; i++)
    {
        result[i] = vector1.Span[i] + vector2.Span[i];
    }
    return new ReadOnlyMemory<float>(result);
}

decimal CosineSimilairty(ReadOnlyMemory<float> vector1, ReadOnlyMemory<float> vector2)
{
    if (vector1.Length != vector2.Length)
        throw new ArgumentException("Vectors must be of the same length");

    float dotProduct = 0;
    float magnitude1 = 0;
    float magnitude2 = 0;

    for (int i = 0; i < vector1.Length; i++)
    {
        dotProduct += vector1.Span[i] * vector2.Span[i];
        magnitude1 += vector1.Span[i] * vector1.Span[i];
        magnitude2 += vector2.Span[i] * vector2.Span[i];
    }

    magnitude1 = (float)Math.Sqrt(magnitude1);
    magnitude2 = (float)Math.Sqrt(magnitude2);

    if (magnitude1 == 0 || magnitude2 == 0)
        return 0;

    return (decimal)(dotProduct / (magnitude1 * magnitude2));
}

static List<Recipe> GetRecipeData()
{

    var recipes = new List<Recipe>();
    var filePath = "data.csv"; // Update with the actual path to your CSV file

    var config = new CsvConfiguration(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true,

    };

    using (var reader = new StreamReader(filePath))
    using (var csv = new CsvReader(reader, config))
    {
        csv.Context.RegisterClassMap<RecipeMap>();
        recipes = csv.GetRecords<Recipe>().ToList();
    }

    return recipes;
}

public class Recipe
{
    [VectorStoreRecordKey]
    public int Id { get; set; }

    [VectorStoreRecordData]
    public string Title { get; set; }

    [VectorStoreRecordData]
    public string Ingredients { get; set; }

    [VectorStoreRecordVector(1024, DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Vector { get; set; }
}

public sealed class RecipeMap : ClassMap<Recipe>
{
    public RecipeMap()
    {
        Map(m => m.Id).Index(0);
        Map(m => m.Title).Index(1);
        Map(m => m.Ingredients).Index(2);
        // Do not map the Vector field
    }
}