/* eslint-disable react-hooks/rules-of-hooks */
import {
    BaseBoxShapeUtil,
    TLBaseShape,
    SvgExportContext,
    useIsEditing,
    useValue,
    Vec,
} from '@tldraw/tldraw'

import { RecordProps, T, TLOnHandleDragHandler } from 'tldraw'
// import CodeMirrorMerge, { CodeMirrorMergeProps } from 'react-codemirror-merge';

import CodeMirror, { EditorView, ChangeSet, EditorState, StateEffect, type ReactCodeMirrorRef, Decoration, WidgetType, DecorationSet, gutter, GutterMarker } from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { javascript } from '@codemirror/lang-javascript';
import { useEffect, useRef, useState, useCallback } from 'react';
import html2canvas from 'html2canvas';
// import * as JsDiff from "diff";
import { acceptChunk, unifiedMergeView, updateOriginalDoc } from '@codemirror/merge';
import { undo } from "@codemirror/commands";
import { RangeSet, StateField } from '@codemirror/state';
import { syntaxTree } from '@codemirror/language';
// import { createTheme } from '@uiw/codemirror-themes';
// const Original = CodeMirrorMerge.Original;
// const Modified = CodeMirrorMerge.Modified;

export type CodeEditorShape = TLBaseShape<
    'code-editor-shape',
    {
        w: number
        h: number
        code: string
        prevCode: string
        res: string
        interpretations: {
            source: {
                startLine: number
                endLine: number
            }
            action: string
            target: {
                startLine: number
                endLine: number
            }
        }
        taskFiles?: { [key: string]: string }
    }
>


const BubbleMenu = ({ onCopy, onPaste, onDelete, position }: any) => {
    return (
        <div
            style={{
                position: 'absolute',
                top: position.top,
                left: position.left,
                zIndex: 999,
                padding: '5px',
                border: '1px solid #ccc',
                borderRadius: '5px',
                boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.1)',
            }}
        >
            <button onClick={onCopy}>Copy</button>
            <button onClick={onPaste}>Paste</button>
            <button onClick={onDelete}>Delete</button>
        </div>
    );
};

const highlightEffect = StateEffect.define<{ from: number; to: number }>({
    map: ({ from, to }, mapping) => ({ from: mapping.mapPos(from), to: mapping.mapPos(to) })
});

const clearHighlightEffect = StateEffect.define<void>({});

const highlightField = StateField.define<DecorationSet>({
    create() {
        return Decoration.none;
    },
    update(highlights, transaction) {
        highlights = highlights.map(transaction.changes);
        for (let effect of transaction.effects) {
            if (effect.is(highlightEffect)) {
                highlights = highlights.update({
                    add: [
                        Decoration.mark({
                            class: 'highlighted-code'
                        }).range(effect.value.from, effect.value.to)
                    ]
                });
            } else if (effect.is(clearHighlightEffect)) {
                highlights = Decoration.none;
            }
        }
        return highlights;
    },
    provide: f => EditorView.decorations.from(f)
});

function applyHighlights(view: EditorView, code: string) {
    const tree = syntaxTree(view.state);
    const effects: StateEffect<unknown>[] = [];

    tree.iterate({
        enter(node) {
            const nodeText = view.state.doc.sliceString(node.from, node.to);
            // console.log('nodeText', nodeText, code)
            if (nodeText === code) {
                effects.push(highlightEffect.of({ from: node.from, to: node.to }));
            }
        }
    });

    if (effects.length > 0) {
        view.dispatch({ effects });
        setTimeout(() => {
            view.dispatch({
                effects: [clearHighlightEffect.of()]
            });
        }, 2000); // Remove highlights after 2 seconds
    }
}

const sourceMarker = new class extends GutterMarker {
    toDOM() { return document.createTextNode("←") }
}

const targetMarker = new class extends GutterMarker {
    toDOM() { return document.createTextNode("→") }
}
const clearDecorationsEffect = StateEffect.define<void>({});

const interpretationEffect = StateEffect.define<{ from: number, to: number, action: string, isSource: boolean, on: boolean }>({
    map: (val, mapping) => ({ from: mapping.mapPos(val.from), to: mapping.mapPos(val.to), action: val.action, isSource: val.isSource, on: val.on }),
})

// Define the state field to manage the gutter markers
const interpretationState = StateField.define<RangeSet<GutterMarker>>({
    create() { return RangeSet.empty },
    update(set, tr) {
        set = set.map(tr.changes);
        for (let e of tr.effects) {
            if (e.is(interpretationEffect)) {
                const marker = e.value.isSource ? sourceMarker : targetMarker;
                if (e.value.on) {
                    for (let line = e.value.from; line <= e.value.to; line++) {
                        set = set.update({ add: [marker.range(line)] });
                    }
                } else {
                    set = set.update({ filter: from => !(from >= e.value.from && from <= e.value.to) });
                }
            } else if (e.is(verticalRectEffect)) {
                const lineCount = e.value.to - e.value.from + 1;
                const marker = new VerticalRectMarker(e.value.height);
                for (let line = e.value.from; line <= e.value.to; line++) {
                    set = set.update({ add: [marker.range(line)] });
                }
            }
        }
        return set;
    }
});

const interpretationGutter = [
    interpretationState,
    gutter({
        class: "cm-interpretation-gutter",
        markers: v => v.state.field(interpretationState),
        initialSpacer: () => sourceMarker
    }),
    EditorView.baseTheme({
        ".cm-interpretation-gutter .cm-gutterElement": {
            color: '#66A3D2 !important',
            fontWeight: 'bolder !important',
            paddingLeft: "5px",
            cursor: "default"
        },
        ".cm-interpretation-gutter .cm-gutterElement:has(::text('←'))": {
            color: "#66A3D2",
            backgroundColor: "rgba(0, 0, 255, 0.1)"
        },
        ".cm-interpretation-gutter .cm-gutterElement:has(::text('→'))": {
            color: "#66A3D2 !important",
            backgroundColor: "rgba(0, 255, 0, 0.1)"
        }
    })
]

class VerticalRectMarker extends GutterMarker {
    constructor(private height: number) {
        super();
    }

    toDOM(view: EditorView) {
        const div = document.createElement("div");
        div.style.height = `${this.height}px`;
        div.style.width = "4px";
        div.style.backgroundColor = "#C9D9E3";
        div.style.position = "relative";
        div.style.left = "50%";
        div.style.transform = "translateX(-50%)";
        return div;
    }
}

// Define the effect to add or remove the marker
const verticalRectEffect = StateEffect.define<{ from: number, to: number, height: number }>({
    map: (val, mapping) => ({
        from: mapping.mapPos(val.from),
        to: mapping.mapPos(val.to),
        height: val.height,
    })
});


function applyInterpretations(view: EditorView, interpretations: any, clearMarkers = true) {
    const effects = [];

    try {
        if (clearMarkers) {
            effects.push(clearDecorationsEffect.of());
            // put every line on: false
            for (let i = 1; i < view.state.doc.lines; i++) {
                effects.push(
                    interpretationEffect.of({
                        from: view.state.doc.line(i).from,
                        to: view.state.doc.line(i).to,
                        action: interpretations.action,
                        isSource: true,
                        on: false
                    }),
                    interpretationEffect.of({
                        from: view.state.doc.line(i).from,
                        to: view.state.doc.line(i).to,
                        action: interpretations.action,
                        isSource: false,
                        on: false
                    })
                );
            }
        }

        if (interpretations.source.startLine && interpretations.source.endLine) {
            effects.push(
                verticalRectEffect.of({
                    from: view.state.doc.line(interpretations.source.startLine).from,
                    to: view.state.doc.line(interpretations.source.endLine).to,
                    height: (interpretations.target.endLine - interpretations.target.startLine + 1) * 14 // Adjust height based on the number of lines
                }),
            );
        }
        if (interpretations.target.startLine && interpretations.target.endLine) {
            effects.push(
                interpretationEffect.of({
                    from: view.state.doc.line(interpretations.target.startLine).from,
                    to: view.state.doc.line(interpretations.target.endLine).to,
                    action: interpretations.action,
                    isSource: false,
                    on: true
                })
            )
        }

        view.dispatch({ effects });
    } catch (e) {
        console.error(e);
    }
}

export class CodeEditorShapeUtil extends BaseBoxShapeUtil<CodeEditorShape> {
    static override type = 'code-editor-shape' as const

    static override props: RecordProps<CodeEditorShape> = {
        w: T.number,
        h: T.number,
        code: T.string,
        prevCode: T.string,
        res: T.string,
        interpretations: T.any,
        taskFiles: T.any,
    }

    getDefaultProps(): CodeEditorShape['props'] {
        return {
            w: 400,
            h: 300,
            code: ``,
            prevCode: '',
            res: '',
            interpretations: {
                source: {
                    startLine: 0,
                    endLine: 0
                },
                action: '',
                target: {
                    startLine: 0,
                    endLine: 0
                }
            },
            taskFiles: {}
        }
    }

    override canEdit = () => true
    override isAspectRatioLocked = () => false
    override canResize = () => true
    override canBind = () => false
    // override canUnmount = () => false
    override hideSelectionBoundsFg = () => true
    override hideSelectionBoundsBg = () => true
    override canScroll = () => true
    override canSnap = () => true


    override component(shape: CodeEditorShape) {
        const isEditing = useIsEditing(shape.id)
        const startPosition = useRef<number | null>(null)
        const [selectionRange, setSelectionRange] = useState<{ anchor: number, head: number } | null>(null)
        const [menuPosition, setMenuPosition] = useState<{ top: number, left: number } | null>(null)


        const codeMirrorRef = useRef<ReactCodeMirrorRef | null>(null)
        const extensions = [
            python(),
            javascript(),
            unifiedMergeView({
                original: shape.props.prevCode,
                syntaxHighlightDeletions: false,
                mergeControls: false
            }),
            interpretationGutter,
            highlightField
        ]


        useEffect(() => {
            const view = codeMirrorRef.current?.view;
            if (view && shape.props.interpretations) {
                applyInterpretations(view, shape.props.interpretations);

                const codeMatches = shape.props.interpretations.action.match(/\[\[CODE:(.*?)\]\]/);
                if (codeMatches?.[1]) {
                    // console.log('CODE MATCHES', codeMatches[1]);
                    applyHighlights(view, codeMatches[1]);
                }
            }
        }, [shape.props.interpretations]);

        useEffect(() => {
            if (codeMirrorRef.current) {
                const view = codeMirrorRef.current?.view
                console.log('shape.props.code', view?.contentHeight, shape.props.h);

                this.editor.updateShape<CodeEditorShape>({
                    id: shape.id,
                    type: 'code-editor-shape',
                    isLocked: false,
                    props: {
                        ...shape.props,
                        h: Math.max(600, Math.min(1100, shape.props.h)) // Ensure reasonable height
                    }
                })

                const codeMirrorView = codeMirrorRef.current?.view as EditorView
                if (codeMirrorView) {
                    codeMirrorView?.dispatch({
                        effects: updateOriginalDoc.of({
                            doc: codeMirrorView.state.toText(shape.props.prevCode),
                            changes: ChangeSet.empty(0),
                        }),
                    });

                    const code = shape.props.code;
                    const prevCode = shape.props.prevCode;
                    if (code.length === prevCode.length + 1 && code.endsWith(' ') && code.slice(0, -1) === prevCode) {
                        undo(codeMirrorView);
                    }
                }

                this.editor.updateShape<CodeEditorShape>({
                    id: shape.id,
                    type: 'code-editor-shape',
                    isLocked: true,
                })
            }
        }, [shape.props.code, shape.props.prevCode])


        useEffect(() => {
            if (shape.props.res) {
                console.log('shape.props.res', shape.props.res);
            }
        }, [shape.props.res]);


        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

        // const handleTouchStart = (e: React.TouchEvent<HTMLDivElement>) => {
        //     const touch = e.touches[0];
        //     const editorView = codeMirrorRef.current?.view;
        //     if (editorView) {
        //         const pos = editorView.posAtCoords({ x: touch.clientX, y: touch.clientY });
        //         if (pos) {
        //             editorView.dispatch({ selection: { anchor: pos, head: pos } });
        //         }
        //         startPosition.current = editorView.posAtCoords({ x: touch.clientX, y: touch.clientY }) || 0;
        //     }
        //     e.stopPropagation();
        //     return;
        // }

        // const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
        //     const touch = e.touches[0];
        //     const editorView = codeMirrorRef.current?.view;
        //     if (editorView) {
        //         const pos = editorView.posAtCoords({ x: touch.clientX, y: touch.clientY });
        //         if (pos && startPosition.current) {
        //             editorView.dispatch({ selection: { anchor: startPosition.current, head: pos } });
        //         } else if (pos) {
        //             editorView.dispatch({ selection: { anchor: pos, head: pos } });
        //             startPosition.current = pos || 0;
        //         }
        //     }
        //     return;
        // }

        // useEffect(() => {
        //     let timeoutId: any;
        //     if (selectionRange) {
        //         const view = codeMirrorRef.current?.view;
        //         const coords = view?.coordsAtPos(selectionRange.anchor);
        //         if (coords) {
        //             setMenuPosition({ top: coords.top + 20, left: coords.left });
        //             timeoutId = setTimeout(() => {
        //                 setSelectionRange(null);
        //                 view?.dispatch({ selection: { anchor: selectionRange.anchor, head: selectionRange.anchor } });
        //             }, 2000);
        //         }
        //     }
        //     return () => clearTimeout(timeoutId);
        // }, [selectionRange]);

        const handleCopy = () => {
            const view = codeMirrorRef.current?.view;
            if (view && selectionRange) {
                const selectedText = view.state.sliceDoc(selectionRange.anchor, selectionRange.head);
                navigator.clipboard.writeText(selectedText);
                setSelectionRange(null);
            }
        };

        const handlePaste = () => {
            navigator.clipboard.readText().then((clipText) => {
                const view = codeMirrorRef.current?.view;
                if (view && selectionRange) {
                    view.dispatch(view.state.replaceSelection(clipText));
                    setSelectionRange(null);
                }
            });
        };

        const handleDelete = () => {
            const view = codeMirrorRef.current?.view;
            if (view && selectionRange) {
                view.dispatch(view.state.replaceSelection(''));
                setSelectionRange(null);
            }
        };

        // const handleTouchEnd = (e: React.TouchEvent<HTMLDivElement>) => {
        //     const view = codeMirrorRef.current?.view;
        //     if (view) {
        //         const selection = view.state.selection.main;
        //         if (selection.from !== selection.to) {
        //             setSelectionRange({ anchor: selection.from, head: selection.to });
        //         }
        //     }
        //     startPosition.current = null;
        //     e.stopPropagation();
        // };

        const handleDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
            e.preventDefault();
            e.stopPropagation();
            return
        };

        return (
            <div
                id={`editor-${shape.id}`}
                onDoubleClick={handleDoubleClick}
                style={{
                    touchAction: isTouchDevice ? 'auto' : 'none',
                    pointerEvents: isEditing ? 'auto' : 'none',
                    alignItems: 'center',
                    position: 'relative',
                    width: `${shape.props.w}px`,
                    height: `${shape.props.h}px`,
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden'
                }}
            >
                <div
                    style={{ 
                        position: 'relative', 
                        zIndex: 1, 
                        width: '100%',
                        height: '60%'
                    }}
                >
                    <CodeMirror
                        ref={codeMirrorRef}
                        value={shape.props.code}
                        style={{
                            fontSize: '16px',
                            border: '1px solid var(--color-panel-contrast)',
                            borderRadius: 'var(--radius-2)',
                            backgroundColor: 'var(--color-background)',
                            width: `${shape.props.w}px`,
                            height: '100%'
                        }}
                        extensions={[...extensions, interpretationGutter]}
                        height="100%"
                        editable={true}
                    />
                </div>
                <div id="result-view" style={{
                    width: '100%',
                    height: '40%',
                    minHeight: '100px',
                    overflow: 'auto',
                    borderTop: '5px solid #ccc',
                    fontFamily: '"Fira Code", "Consolas", "Monaco", monospace',
                    backgroundColor: 'rgb(248 248 248)',
                    color: '#000',
                    padding: '20px',
                    boxSizing: 'border-box',
                    zIndex: 10,
                    whiteSpace: 'pre-wrap',
                    position: 'relative'
                }}
                    contentEditable={false}
                    tabIndex={-1}
                    onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); return; }}
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); return; }}
                    onTouchStart={(e) => { e.preventDefault(); e.stopPropagation(); return; }}
                >
                    <div style={{ 
                        width: '100%',
                        height: '100%',
                        overflow: 'visible'
                    }} dangerouslySetInnerHTML={{ __html: shape.props.res }}></div>
                </div>
                {selectionRange && menuPosition && (
                    <BubbleMenu
                        onCopy={handleCopy}
                        onPaste={handlePaste}
                        onDelete={handleDelete}
                        position={menuPosition}
                    />
                )}
            </div>
        )
    }

    override async toSvg(shape: CodeEditorShape, _ctx: SvgExportContext): Promise<React.ReactElement | null> {
        const element = document.getElementById(`editor-${shape.id}`) as HTMLElement;
        if (!element) return null;

        // Wait for any pending renders to complete
        await new Promise(resolve => setTimeout(resolve, 500));

        // Find the size of the entire element including any overflow content
        const computedStyle = window.getComputedStyle(element);
        const resultView = element.querySelector('#result-view') as HTMLElement;
        const codeEditor = element.querySelector('.cm-editor') as HTMLElement;
        
        // Make sure the cloned element captures everything
        const screenShot = await html2canvas(element, {
            useCORS: true,
            scale: 2,
            logging: false,
            backgroundColor: null,
            allowTaint: true,
            windowWidth: shape.props.w,
            windowHeight: shape.props.h,
            scrollX: 0,
            scrollY: 0,
            width: shape.props.w,
            height: shape.props.h,
            ignoreElements: (el) => false, // Don't ignore any elements
            onclone: (clonedDoc) => {
                const clonedElement = clonedDoc.getElementById(`editor-${shape.id}`);
                if (clonedElement) {
                    // Make sure dimensions are set correctly
                    clonedElement.style.width = `${shape.props.w}px`;
                    clonedElement.style.height = `${shape.props.h}px`;
                    
                    // Fix code editor visibility
                    const clonedEditor = clonedElement.querySelector('.cm-editor');
                    if (clonedEditor) {
                        (clonedEditor as HTMLElement).style.height = '100%';
                        (clonedEditor as HTMLElement).style.visibility = 'visible';
                        (clonedEditor as HTMLElement).style.display = 'block';
                    }
                    
                    // Fix result view visibility and ensure overflow content is visible
                    const clonedResultView = clonedElement.querySelector('#result-view');
                    if (clonedResultView) {
                        (clonedResultView as HTMLElement).style.height = '40%';
                        (clonedResultView as HTMLElement).style.visibility = 'visible';
                        (clonedResultView as HTMLElement).style.display = 'block';
                        
                        // Make sure charts/visualization are visible
                        const resultContent = clonedResultView.querySelector('div');
                        if (resultContent) {
                            (resultContent as HTMLElement).style.position = 'relative';
                            (resultContent as HTMLElement).style.visibility = 'visible';
                            (resultContent as HTMLElement).style.overflow = 'visible';
                        }
                    }
                }
            }
        }).then(function (canvas) {
            return canvas.toDataURL('image/png', 1.0);
        });

        return (
            <svg width={shape.props.w} height={shape.props.h} xmlns="http://www.w3.org/2000/svg">
                <image
                    href={screenShot}
                    width={shape.props.w}
                    height={shape.props.h}
                    preserveAspectRatio="none"
                />
            </svg>
        );
    }

    override onDoubleClick = (shape: CodeEditorShape) => {
        return;
    }

    override onRotateStart = (shape: CodeEditorShape) => {
        return;
    }

    // override onResizeStart = (shape: CodeEditorShape) => {
    //     return;
    // }

    override onHandleDrag = (shape: CodeEditorShape) => {
        return;
    }

    indicator(shape: CodeEditorShape) {
        return <rect width={shape.props.w} height={shape.props.h} />
    }
}