import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";
import { Langfuse } from "langfuse";

// Constants for action names and input names
const ACTION_OPENED = "opened";
const ACTION_SYNCHRONIZE = "synchronize";
const ACTION_LABELED = "labeled";
const INPUT_GITHUB_TOKEN = "GITHUB_TOKEN";
const INPUT_OPENAI_API_KEY = "OPENAI_API_KEY";
const INPUT_OPENAI_API_MODEL = "OPENAI_API_MODEL";
const INPUT_EXCLUDE = "exclude";
const INPUT_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY";
const INPUT_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY";
const INPUT_REVIEW_LABEL = "REVIEW_LABEL";

// TypeScript interfaces for GitHub event data
interface Repository {
  owner: {
    login: string;
  };
  name: string;
}

interface PullRequestEvent {
  action: string;
  number: number;
  repository: Repository;
  before?: string;
  after?: string;
}

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
  labels: string[];
}

interface AIReview {
  lineNumber: string;
  reviewComment: string;
}

interface Comment {
  body: string;
  path: string;
  line: number;
}

// Initialize Langfuse
const langfuse = new Langfuse({
  requestTimeout: 10000,
  enabled: Boolean(core.getInput(INPUT_LANGFUSE_SECRET_KEY)),
  secretKey: core.getInput(INPUT_LANGFUSE_SECRET_KEY),
  publicKey: core.getInput(INPUT_LANGFUSE_PUBLIC_KEY),
});

// Handle Langfuse errors
langfuse.on("error", (error) => {
  console.error("Langfuse Error:", error);
});

// Optionally enable debugging
// langfuse.debug();

// Initialize Octokit with GitHub Token
const octokit = new Octokit({
  auth: core.getInput(INPUT_GITHUB_TOKEN, { required: true }),
});

// Initialize OpenAI with API Key
const openai = new OpenAI({
  apiKey: core.getInput(INPUT_OPENAI_API_KEY, { required: true }),
});

// OpenAI query configuration
const OPENAI_QUERY_CONFIG = {
  model: core.getInput(INPUT_OPENAI_API_MODEL, { required: true }),
  temperature: 0.2,
  max_tokens: 700,
  top_p: 1,
  frequency_penalty: 0,
  presence_penalty: 0,
};

// Function to get PR details from event data
async function getPRDetails(eventData: PullRequestEvent): Promise<PRDetails> {
  try {
    const prResponse = await octokit.pulls.get({
      owner: eventData.repository.owner.login,
      repo: eventData.repository.name,
      pull_number: eventData.number,
    });

    const labels = prResponse.data.labels
      ? prResponse.data.labels.map((l: { name?: string }) => l.name ?? "")
      : [];

    return {
      owner: eventData.repository.owner.login,
      repo: eventData.repository.name,
      pull_number: eventData.number,
      title: prResponse.data.title ?? "",
      description: prResponse.data.body ?? "",
      labels,
    };
  } catch (error) {
    throw new Error(`Failed to get PR details: ${(error as Error).message}`);
  }
}

// Function to get the diff of a pull request
async function getDiff(
  owner: string,
  repo: string,
  pull_number: number,
): Promise<string> {
  try {
    const response = await octokit.pulls.get({
      owner,
      repo,
      pull_number,
      mediaType: { format: "diff" },
    });

    // Explicitly type response.data as string
    return String(response.data);
  } catch (error) {
    throw new Error(`Failed to get PR diff: ${(error as Error).message}`);
  }
}

// Function to analyze code changes using AI
async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails,
  trace: any, // Adjust the type as per Langfuse's TypeScript definitions
): Promise<Comment[]> {
  const comments: Comment[] = [];

  // Process each file concurrently
  await Promise.all(
    parsedDiff.map(async (file) => {
      if (file.to === "/dev/null") return; // Ignore deleted files

      // Process each chunk in the file concurrently
      const fileComments = await Promise.all(
        file.chunks.map(async (chunk) => {
          const prompt = createPrompt(file, chunk, prDetails);

          // Create a generation in Langfuse
          const generation = trace.generation({
            name: "openai-chat-completion",
            model: OPENAI_QUERY_CONFIG.model,
            modelParameters: {
              temperature: OPENAI_QUERY_CONFIG.temperature,
              maxTokens: OPENAI_QUERY_CONFIG.max_tokens,
              top_p: OPENAI_QUERY_CONFIG.top_p,
              frequency_penalty: OPENAI_QUERY_CONFIG.frequency_penalty,
              presence_penalty: OPENAI_QUERY_CONFIG.presence_penalty,
            },
            input: prompt,
          });

          const aiResponses = await getAIResponse(prompt, generation);
          if (aiResponses) {
            generation.end({
              output: aiResponses,
            });
            return createComments(file, aiResponses);
          } else {
            generation.end({
              output: null,
            });
            return [];
          }
        }),
      );

      // Flatten and add to comments
      fileComments.forEach((commentArray) => {
        comments.push(...commentArray);
      });
    }),
  );

  return comments;
}

// Function to create AI prompt
function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in the following JSON format: {"reviews": [{"lineNumber": <line_number>, "reviewComment": "<review comment>"}]}
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Focus exclusively on functional issues, potential bugs, and performance improvements.
- For imports: ONLY comment if the import statement would cause a definitive runtime error.
- Key areas to review:
  * Error handling and exception cases
  * Resource management (file handles, memory, etc.)
  * Security concerns
  * Performance bottlenecks
  * File system operations
  * Process management
  * Standard compliance issues
- Use the given description only for the overall context and only comment on the code.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "${
    file.to
  }" and take the pull request title and description into account when writing the response.

Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`
`;
}

// Function to get AI response
async function getAIResponse(
  prompt: string,
  generation: any,
): Promise<AIReview[] | null> {
  try {
    const response = await openai.chat.completions.create({
      ...OPENAI_QUERY_CONFIG,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
    });

    const aiContent = response.choices[0].message?.content?.trim();
    if (!aiContent) {
      console.warn("AI response is empty.");
      return null;
    }

    console.log("AI response:", aiContent);

    const parsed = JSON.parse(aiContent);
    return parsed.reviews as AIReview[];
  } catch (error) {
    console.error("Error fetching AI response:", (error as Error).message);
    return null;
  }
}

// Function to create comments from AI responses
function createComments(file: File, aiResponses: AIReview[]): Comment[] {
  if (!file.to) return [];

  return aiResponses.map((aiResponse) => ({
    body: aiResponse.reviewComment,
    path: file.to || "",
    line: Number(aiResponse.lineNumber),
  }));
}

// Function to create review comments on GitHub
async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Comment[],
): Promise<void> {
  try {
    await octokit.pulls.createReview({
      owner,
      repo,
      pull_number,
      comments,
      event: "COMMENT",
    });
    console.log(`Created ${comments.length} review comment(s).`);
  } catch (error) {
    throw new Error(
      `Failed to create review comments: ${(error as Error).message}`,
    );
  }
}

// Main function orchestrating the workflow
async function main() {
  let trace;
  try {
    const eventPath = process.env.GITHUB_EVENT_PATH;
    if (!eventPath) {
      throw new Error("GITHUB_EVENT_PATH is not defined.");
    }

    const eventDataRaw = readFileSync(eventPath, "utf8");
    const eventData: PullRequestEvent = JSON.parse(eventDataRaw);

    if (!eventData.action) {
      throw new Error("Event action is not defined.");
    }

    // Only handle 'opened', 'synchronize' and 'labeled' actions
    if (
      ![ACTION_OPENED, ACTION_SYNCHRONIZE, ACTION_LABELED].includes(
        eventData.action,
      )
    ) {
      console.log(`Unsupported event action: ${eventData.action}`);
      return;
    }

    // Get PR details
    const prDetails = await getPRDetails(eventData);

    const requiredLabel = core.getInput(INPUT_REVIEW_LABEL);
    if (requiredLabel && !prDetails.labels.includes(requiredLabel)) {
      console.log(
        `Required label '${requiredLabel}' not found on PR. Skipping analysis.`,
      );
      return;
    }

    // Initialize trace
    trace = langfuse.trace({
      name: "github-action-pr-review",
      userId: prDetails.owner, // You can adjust this to a more appropriate identifier
      metadata: { repo: prDetails.repo, pull_number: prDetails.pull_number },
      tags: ["github-action"],
    });

    // Get diff based on action
    let diff: string;
    if (
      eventData.action === ACTION_OPENED ||
      eventData.action === ACTION_LABELED
    ) {
      diff = await getDiff(
        prDetails.owner,
        prDetails.repo,
        prDetails.pull_number,
      );
    } else if (eventData.action === ACTION_SYNCHRONIZE) {
      if (!eventData.before || !eventData.after) {
        throw new Error(
          "Both 'before' and 'after' SHAs are required for synchronize action.",
        );
      }

      const response = await octokit.repos.compareCommits({
        headers: {
          accept: "application/vnd.github.v3.diff",
        },
        owner: prDetails.owner,
        repo: prDetails.repo,
        base: eventData.before,
        head: eventData.after,
      });

      diff = String(response.data);
    } else {
      // This else is redundant due to the earlier check but kept for safety
      console.log("No diff found for unsupported action.");
      return;
    }

    if (!diff) {
      console.log("No diff found.");
      return;
    }

    // Parse the diff
    const parsedDiff = parseDiff(diff);

    // Get exclude patterns
    const excludePatternsInput = core.getInput(INPUT_EXCLUDE);
    const excludePatterns = excludePatternsInput
      ? excludePatternsInput.split(",").map((pattern) => pattern.trim())
      : [];

    // Filter out excluded files
    const filteredDiff = parsedDiff.filter((file) => {
      const filePath = file.to ?? "";
      return !excludePatterns.some((pattern) => minimatch(filePath, pattern));
    });

    if (filteredDiff.length === 0) {
      console.log("No files to analyze after applying exclude patterns.");
      return;
    }

    // Analyze code to get comments
    const comments = await analyzeCode(filteredDiff, prDetails, trace);

    if (comments.length > 0) {
      await createReviewComment(
        prDetails.owner,
        prDetails.repo,
        prDetails.pull_number,
        comments,
      );
    } else {
      console.log("No comments to post.");
    }
  } catch (error) {
    console.error("Error in main execution:", (error as Error).message);
    core.setFailed((error as Error).message);
    process.exit(1);
  } finally {
    // @ts-ignore
    if (langfuse.enabled) {
      await langfuse.shutdownAsync();
    }
  }
}

// Execute the main function
main();
