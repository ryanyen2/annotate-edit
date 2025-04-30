import {
    Editor, createShapeId, TLBaseShape, TLImageShape, TLTextShape,
    AssetRecordType, type TLShapeId
} from '@tldraw/tldraw'
import { getSelectionAsText } from './getSelectionAsText'
import { CodeEditorShape } from '../CodeEditorShape/CodeEditorShape'
//@ts-ignore
import * as xPython from '@x-python/core'
import { downloadDataURLAsFile } from './downloadDataUrlAsFile';
import { PreviewShape } from '../PreviewShape/PreviewShape';
import { userStudyTasks } from './tasks';
// import { writeFileSync } from 'fs';
// import { promisify } from 'util';

// const writeFileSyncAsync = promisify(writeFileSync)


export type CodeExecResultType = "image" | "text" | "table";
export interface CodeExecReturnValue {
    result: any | null;
    error: string | null;
    stdout: string | null;
    stderr: string | null;
    type: CodeExecResultType;
    files?: { [key: string]: string };
}

const packages = {
    official: ["pandas", "matplotlib", "numpy", "scipy"],
    micropip: ["seaborn", "scikit-learn"],
};
const matplotlibCode = `import os
import base64
from io import BytesIO\n
# Set this _before_ importing matplotlib
os.environ['MPLBACKEND'] = 'AGG'\n
import matplotlib.pyplot as plt\n
# Patch
def ensure_matplotlib_patch():
  _old_show = plt.show\n
  def show():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Encode to a base64 str
    img = 'data:image/png;base64,' + \\
    base64.b64encode(buf.read()).decode('utf-8')
    # Write to stdout
    print(img)
    plt.clf()\n
  plt.show = show\n\n
ensure_matplotlib_patch()\n
`;

const anotherPatchPrefix = `import os
from matplotlib import pyplot as plt
import io
import base64
import js
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
import ast


class Dud:

    def __init__(self, *args, **kwargs) -> None:
        return

    def __getattr__(self, __name: str):
        return Dud


js.document = Dud()
`
const anotherPatchSuffix = `

# Print base64 string to stdout
bytes_io = io.BytesIO()

plt.savefig(bytes_io, format='jpg')

bytes_io.seek(0)

base64_encoded_spectrogram = base64.b64encode(bytes_io.read())

print('data:image/jpg;base64,' + base64_encoded_spectrogram.decode('utf-8'))
`

function tableToHtml(output: string, id: TLShapeId, className: string): string {
    const lines = output.split("\n");
    let html = `<style>
        .${className} {
            width: 100%;
            text-align: left;
            color: #718096;
            border-collapse: collapse;
            border-radius: 15px;
            overflow: hidden;
        }

        .${className} thead {
            font-size: 12px;
            color: #4a5568;
            background-color: #f7fafc;
        }

        .${className} thead th {
            padding: 12px 15px;
        }

        .${className} tbody {
            background-color: #fff;
        }

        .${className} tbody tr {
            color: #718096;
            border-bottom: 1px solid #e2e8f0;
        }

        .${className} tbody tr:hover {
            background-color: #f7fafc;
        }

        .${className} tbody td {
            padding: 12px 15px;
        }
        </style>`;

    html += `<table id="${id}" class="${className}">`;

    // Header, add a blank cell to the start
    const headers = lines[0].split(/\s{2,}/);
    html += "<thead>";
    html += "<tr>";
    html += "<th ></th>";
    for (const header of headers) {
        html += `<th >${header}</th>`;
    }
    html += "</tr>";

    // Rows
    html += "</thead>";
    html += "<tbody>";
    for (let i = 1; i < lines.length; i++) {
        const columns = lines[i].split(/\s{2,}/);
        html += "<tr>";
        for (const column of columns) {
            html += `<td>${column}</td>`;
        }
        html += "</tr>";
    }
    html += "</tbody>";

    html += "</table>";

    return html;
}

const codeRefactoring = async (code: string): Promise<string> => {
    let lines = code.split("\n");
    let lastLineIndex = lines.length - 1;

    // Find the last non-empty line
    while (lastLineIndex >= 0 && lines[lastLineIndex].trim() === '') {
        lastLineIndex--;
    }

    if (lastLineIndex < 0) {
        return code;
    }

    const lastLine = lines[lastLineIndex];

    if (
        code.includes("matplotlib") ||
        code.includes("plt.show") ||
        code.includes("sns")
    ) {
        // return `${matplotlibCode}\n${code}`;
        return `${anotherPatchPrefix}\n${code}\n${anotherPatchSuffix}`;
        // return code
    }
    else if (lastLine.includes("df.") && !lastLine.includes("print")) {
        return code.replace(lastLine, `print(${lastLine})`);
    }
    else {
        return code;
    }
};

const installExtraPackages = async (code: string) => {
    if (code.includes("import seaborn") || code.includes("sns")) {
        await xPython.install(["seaborn"]);
    } else if (code.includes("import sklearn") || code.includes("sklearn")) {
        await xPython.install(["scikit-learn"]);
    } else if (code.includes("import keras") || code.includes("keras")) {
        await xPython.install(["keras"]);
    } else if (code.includes("import nltk") || code.includes("nltk")) {
        await xPython.install(["nltk"]);
    } else if (code.includes("import spacy") || code.includes("spacy")) {
        await xPython.install(["spacy"]);
    }
};

const isTable = (result: string): boolean => {
    // detect if the result(string) is a table, and parse it to html
    // example string that contains table: "sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n0                5.1               3.5                1.4               0.2\n1                4.9               3.0                1.4               0.2\n2                4.7               3.2                1.3               0.2\n3                4.6               3.1                1.5               0.2\n4                5.0               3.6                1.4               0.2"
    const lines = result.split("\n");
    if (lines.length <= 1) {
        return false;
    }
    const columnsLine1 = lines[0].split(/\s{2,}/);
    const columnsLine2 = lines[1].split(/\s{2,}/);
    const columnsLine3 = lines[2].split(/\s{2,}/);

    return (
        columnsLine1.length === columnsLine2.length - 1 &&
        columnsLine2.length === columnsLine3.length
    );
};

const decideExecResultType = (result: any) => {
    if (result && result.toString().includes("data:image/png;base64")) {
        return "image";
    } else if (result && isTable(result)) {
        return "table";
    } else {
        return "text";
    }
};

export async function executeCode(editor: Editor, codeShapeId: TLShapeId) {
    await xPython.init();

    // Get the latest code for pipeline.py
    let pipelineCodeShape = editor.getShape<CodeEditorShape>(codeShapeId);
    if (!pipelineCodeShape) {
        pipelineCodeShape = editor.getCurrentPageShapes().find(shape => shape.type === 'code-editor-shape') as CodeEditorShape;
    }
    if (!pipelineCodeShape || pipelineCodeShape.props.code === '') {
        throw Error('No code to execute.');
    }

    // Extract task ID safely
    const taskId = '4-2';
    const taskPrefix = taskId.split('-')[0];

    const currentCode = pipelineCodeShape.props.code;
    const relatedFileCode = userStudyTasks.find(task => task.id === taskId)?.starterCode || '';
    
    let combinedCode = '';
    
    // Special handling for CHI paper analysis tasks (4-1 and 4-2)
    if (taskPrefix === '4') {
        try {
            // Load the CHI data from the public directory as CSV
            const chiDataResponse = await fetch('/data/CHI_2025_program_cleaned.csv');
            if (!chiDataResponse.ok) {
                throw new Error(`Failed to fetch CHI data: ${chiDataResponse.statusText}`);
            }
            const csvText = await chiDataResponse.text();
            
            // Convert CSV text to a string that can be safely embedded in Python
            // Escape any triple quotes and newlines to preserve CSV format
            const escapedCsvText = csvText
                .replace(/"""/g, '\\"\\"\\"')
                .replace(/\n/g, '\\n');
            
            // Create Python code that directly initializes the DataFrame with the CSV data
            combinedCode = `
import pandas as pd
import io

# Load CHI data from CSV string
csv_data = """${escapedCsvText}"""
df = pd.read_csv(io.StringIO(csv_data))

# User code starts here
${currentCode}
`;
        } catch (error) {
            console.error("Failed to load CHI data:", error);
            
            // Create a minimal sample dataset for testing/development
            const sampleCsv = `id,title,trackId,abstract,award
1,"Sample Paper 1",track1,"This is a sample abstract for testing.",BEST_PAPER
2,"Sample Paper 2",track2,"Another sample abstract for testing.",
3,"Sample Paper 3",track1,"Third sample abstract about HCI.",HONORABLE_MENTION`;
            
            combinedCode = `
import pandas as pd
import io

# Using sample data since actual data couldn't be loaded
csv_data = """${sampleCsv}"""
df = pd.read_csv(io.StringIO(csv_data))
print("Note: Using sample data instead of actual CHI data")

# User code starts here
${currentCode}
`;
        }
    } else {
        combinedCode = `${currentCode}\n\n${relatedFileCode}`;
    }
    
    console.log('Combined code length:', combinedCode);
    
    // Refactor and execute the combined code
    const allSelectedCode = await codeRefactoring(combinedCode);
    await installExtraPackages(allSelectedCode);
    const { result: code } = await xPython.format({ code: allSelectedCode });

    const { error, stdout, stderr } = (await xPython.exec({ code })) as CodeExecReturnValue;

    if (error || !stdout) {
        throw Error(error || 'No output from the code.');
    }

    console.log(stdout);

    // const resultType = decideExecResultType(stdout) as CodeExecResultType;
    const resultType = 'text';
    let htmlResult = '';

    // Process the output
    const images = stdout.match(/data:image\/jpg;base64,[^"]+/g) || [];
    let nonImageContent = stdout;
    const imageHtml = images.map(image => `<img src="${image}" alt="image" width="400">`).join('');
    images.forEach(image => {
        nonImageContent = nonImageContent.replace(image, '');
    });
    const nonImageHtml = `<pre>${nonImageContent.trim()}</pre>`;
    htmlResult = `${nonImageHtml}${imageHtml}`;

    if (htmlResult === '') {
        throw Error('No result to display.');
    }

    editor.updateShape<CodeEditorShape>({
        id: pipelineCodeShape.id,
        type: 'code-editor-shape',
        isLocked: false,
        props: {
            ...pipelineCodeShape.props,
            res: htmlResult,
        },
    });

    editor.updateShape<CodeEditorShape>({
        id: pipelineCodeShape.id,
        type: 'code-editor-shape',
        isLocked: true,
    });

    return htmlResult;
}