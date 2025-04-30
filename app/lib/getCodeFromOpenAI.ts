import { CodeEditorShape } from '../CodeEditorShape/CodeEditorShape'
import {
    OPENAI_MAKE_CODE_PROMPT,
    OPENAI_USER_MAKE_CODE_PROMPT,
    OPENAI_EDIT_PARTIAL_CODE_PROMPT,
    OPENAI_USER_EDIT_PARTIAL_CODE_PROMPT,
} from '../prompt'

export async function getCodeFromOpenAI({
    interpretation,
    image,
    apiKey,
    text,
    grid,
    previousCodeEditors = [],
    // combinedCode,
    intended_edit,
}: {
    interpretation: string
    image: string
    apiKey: string
    text: string
    grid?: {
        color: string
        size: number
        labels: boolean
    }
    previousCodeEditors?: CodeEditorShape[]
    // combinedCode?: string
    intended_edit?: string
}) {
    if (!apiKey) throw Error('You need to provide an API key (sorry)')



    const messages: GPT4oCompletionRequest['messages'] = [
        {
            role: 'system',
            content: intended_edit?.length ? OPENAI_EDIT_PARTIAL_CODE_PROMPT : OPENAI_MAKE_CODE_PROMPT,
        },
        {
            role: 'user',
            content: []
        },
    ]

    const userContent = messages[1].content as Exclude<MessageContent, string>

    userContent.push({
        type: 'text',
        text: intended_edit?.length ? OPENAI_USER_EDIT_PARTIAL_CODE_PROMPT : OPENAI_USER_MAKE_CODE_PROMPT,
    })

    // Add the image
    userContent.push({
        type: 'image_url',
        image_url: {
            url: image,
            detail: 'high',
        },
    })



    userContent.push({
        type: 'text',
        text: `If users circled some words from the output area (below the code editor) and crossed out, it means user want to bypass those words (e.g., "used", "provide", "increasingly") when preprocessing the text, add a line of code to exclude "used" and "provide" from the text.
        For example, if user circled "used", "increasingly" and "provide" and crossed out, and already have a list of words, you should add:
        filtered_words = ["used", "provide", "increasingly"]
        words = [word for word in words if word not in filtered_words]
        `,
    })


    userContent.push({
        type: 'text',
        text: `If and only if users asekd to extract countries out from the data, and draw a visualization about barchart showing the top countries of award winning, you can change the code to (no need import or data loading):
def extract_countries(authors_str):
    try:
        authors = ast.literal_eval(authors_str)
        countries = set()
        for author in authors:
            for aff in author.get('affiliations', []):
                country = aff.get('country')
                if country:
                    countries.add(country)
        return list(countries)
    except (ValueError, SyntaxError):
        return []


df_expanded = df.copy()
df_expanded['Country'] = df_expanded['authors'].apply(extract_countries)
df_expanded = df_expanded.explode('Country').dropna(subset=['Country'])

country_counts = (
    df_expanded
      .groupby('Country')
      .size()
      .reset_index(name='Count')
      .sort_values('Count', ascending=False)
      .head(10)
)

plt.figure(figsize=(12, 10))
sns.barplot(
    data=country_counts,
    y='Country',
    x='Count',
    palette='viridis'
)
plt.title('Top 10 Countries by Award-Winning Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Country')
plt.tight_layout()
plt.show()`,
    })

    userContent.push({
        type: 'text',
        text: `If and only if user add a legned name award type beside the visualization of top 20 award winning country, please then edits the code to create pivot table then visualize, for examlpe:
counts = (
    df_expanded
    .groupby(['Country', 'award'])
    .size()
    .reset_index(name='Count')
)

pivot = counts.pivot(index='Country', columns='award', values='Count').fillna(0)

pivot['Total'] = pivot.sum(axis=1)
top10 = pivot.sort_values('Total', ascending=False).head(10).drop(columns='Total')

award_types = top10.columns.tolist()
countries = top10.index.tolist()

fig, ax = plt.subplots(figsize=(12, 10))
left = [0] * len(countries)
for award in award_types:
    values = top10[award].values
    ax.barh(countries, values, left=left, label=award)
    left = [l + v for l, v in zip(left, values)]

ax.set_xlabel('Number of Award-winning Papers')
ax.set_ylabel('Country')
ax.set_title('Distribution of Awards by Country (Top 10)')
ax.legend(title='Award Type', loc='lower right')
plt.tight_layout()
plt.show()`,
    })



    if (interpretation) {
        userContent.push({
            type: 'text',
            text: `The user specified following action to take: "${interpretation}"`,
        })
    }

    // Add the strings of text
    if (text) {
        userContent.push({
            type: 'text',
            text: `Here's a list of text that we found in the annotations:\n${text}`,
        })
    }

    if (grid) {
        userContent.push({
            type: 'text',
            text: `The user have a ${grid.color} grid overlaid on top. Each cell of the grid is ${grid.size}x${grid.size}px.`,
        })
    }

    // Add the previous previews code
    for (let i = 0; i < previousCodeEditors.length; i++) {
        const preview = previousCodeEditors[i]
        userContent.push({
            type: 'text',
            text: `The users also included the code in the code editor (DO NOT add import statements or import data, which is already included and preloaded, DO NOT edit code user did not specify in the sketch):\n${preview.props.code}`,
        })
    }
    // if (combinedCode) {
    //     userContent.push({
    //         type: 'text',
    //         text: `The users also included the code in the code editor, just focus on modifying the pipeline.py file:\n${combinedCode}`,
    //     })
    // }

    if (intended_edit?.length) {
        userContent.push({
            type: 'text',
            text: `The user intended to edit the code to: "${intended_edit}"`,
        })
    }

    // Prompt the theme
    // userContent.push({
    // 	type: 'text',
    // 	text: `Please make your result use the ${theme} theme.`,
    // })

    const body: GPT4oCompletionRequest = {
        model: 'gpt-4o',
        max_tokens: 4096,
        temperature: 0,
        messages,
        seed: 42,
        n: 1,
    }

    let json = null

    try {
        const resp = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify(body),
        })
        json = await resp.json()
    } catch (e: any) {
        throw Error(`Could not contact OpenAI: ${e.message}`)
    }

    return json
}

type MessageContent =
    | string
    | (
        | string
        | {
            type: 'image_url'
            image_url:
            | string
            | {
                url: string
                detail: 'low' | 'high' | 'auto'
            }
        }
        | {
            type: 'text'
            text: string
        }
    )[]

export type GPT4oCompletionRequest = {
    model: 'gpt-4o'
    messages: {
        role: 'system' | 'user' | 'assistant' | 'function'
        content: MessageContent
        name?: string | undefined
    }[]
    functions?: any[] | undefined
    function_call?: any | undefined
    stream?: boolean | undefined
    temperature?: number | undefined
    top_p?: number | undefined
    max_tokens?: number | undefined
    n?: number | undefined
    best_of?: number | undefined
    frequency_penalty?: number | undefined
    presence_penalty?: number | undefined
    seed?: number | undefined
    logit_bias?:
    | {
        [x: string]: number
    }
    | undefined
    stop?: (string[] | string) | undefined
}
