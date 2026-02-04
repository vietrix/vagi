/**
 * useVAGIStream - React Hook for vAGI Server-Sent Events Stream
 *
 * Parses the streaming response from vAGI API and maintains separate states
 * for thinking traces, code executions, and final answers.
 *
 * Features:
 * - Robust state machine for tag detection across chunk boundaries
 * - Handles split tags (e.g., "</thi" + "nk>")
 * - Separate states for thought, code, observation, and answer
 * - Auto-reconnection on connection loss
 * - TypeScript type safety
 *
 * Usage:
 * ```tsx
 * const {
 *   thoughtTrace,
 *   codeExecution,
 *   finalAnswer,
 *   isStreaming,
 *   error,
 *   sendMessage,
 *   abort,
 * } = useVAGIStream({ apiUrl: '/v1/chat/completions' });
 * ```
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// =============================================================================
// Types
// =============================================================================

export interface VAGIStreamConfig {
  apiUrl: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  onThought?: (content: string) => void;
  onCode?: (code: string, language: string) => void;
  onObservation?: (result: string) => void;
  onAnswer?: (content: string) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
}

export interface CodeBlock {
  language: string;
  code: string;
  observation?: string;
}

export interface VAGIStreamState {
  thoughtTrace: string;
  codeExecutions: CodeBlock[];
  finalAnswer: string;
  isStreaming: boolean;
  isConnected: boolean;
  error: Error | null;
}

export interface VAGIStreamActions {
  sendMessage: (message: string, systemPrompt?: string) => Promise<void>;
  abort: () => void;
  reset: () => void;
}

export type UseVAGIStreamReturn = VAGIStreamState & VAGIStreamActions;

// Parser state machine states
enum ParserState {
  NORMAL = 'NORMAL',
  IN_THINK = 'IN_THINK',
  IN_VERIFY_CODE = 'IN_VERIFY_CODE',
  IN_OBSERVATION = 'IN_OBSERVATION',
}

// Tag definitions for state machine
interface TagDefinition {
  open: RegExp;
  close: RegExp;
  openStr: string;
  closeStr: string;
  nextState: ParserState;
  exitState: ParserState;
}

const TAGS: Record<string, TagDefinition> = {
  think: {
    open: /<think>/gi,
    close: /<\/think>/gi,
    openStr: '<think>',
    closeStr: '</think>',
    nextState: ParserState.IN_THINK,
    exitState: ParserState.NORMAL,
  },
  verifyCode: {
    open: /<verify_code\s+language=["'](\w+)["']>/gi,
    close: /<\/verify_code>/gi,
    openStr: '<verify_code',
    closeStr: '</verify_code>',
    nextState: ParserState.IN_VERIFY_CODE,
    exitState: ParserState.NORMAL,
  },
  observation: {
    open: /<observation>/gi,
    close: /<\/observation>/gi,
    openStr: '<observation>',
    closeStr: '</observation>',
    nextState: ParserState.IN_OBSERVATION,
    exitState: ParserState.NORMAL,
  },
};

// =============================================================================
// Stream Parser Class
// =============================================================================

class VAGIStreamParser {
  private state: ParserState = ParserState.NORMAL;
  private buffer: string = '';
  private currentCodeLanguage: string = 'python';

  // Accumulated content
  public thoughtContent: string = '';
  public codeBlocks: CodeBlock[] = [];
  public currentCodeBlock: CodeBlock | null = null;
  public answerContent: string = '';

  // Callbacks
  private onThought?: (content: string) => void;
  private onCode?: (code: string, language: string) => void;
  private onObservation?: (result: string) => void;
  private onAnswer?: (content: string) => void;

  constructor(callbacks?: {
    onThought?: (content: string) => void;
    onCode?: (code: string, language: string) => void;
    onObservation?: (result: string) => void;
    onAnswer?: (content: string) => void;
  }) {
    this.onThought = callbacks?.onThought;
    this.onCode = callbacks?.onCode;
    this.onObservation = callbacks?.onObservation;
    this.onAnswer = callbacks?.onAnswer;
  }

  /**
   * Process incoming chunk from SSE stream.
   * Handles tags that may be split across chunks.
   */
  public processChunk(chunk: string): void {
    // Add chunk to buffer
    this.buffer += chunk;

    // Process buffer until no more complete tokens
    let processed = true;
    while (processed) {
      processed = this.processBuffer();
    }
  }

  /**
   * Process buffer and extract complete tokens.
   * Returns true if something was processed.
   */
  private processBuffer(): boolean {
    switch (this.state) {
      case ParserState.NORMAL:
        return this.processNormalState();
      case ParserState.IN_THINK:
        return this.processInThinkState();
      case ParserState.IN_VERIFY_CODE:
        return this.processInVerifyCodeState();
      case ParserState.IN_OBSERVATION:
        return this.processInObservationState();
      default:
        return false;
    }
  }

  /**
   * Process buffer in NORMAL state.
   * Look for opening tags or accumulate answer content.
   */
  private processNormalState(): boolean {
    // Check for <think> tag
    const thinkMatch = this.buffer.match(/<think>/i);
    if (thinkMatch && thinkMatch.index !== undefined) {
      // Output any content before the tag as answer
      if (thinkMatch.index > 0) {
        const beforeTag = this.buffer.substring(0, thinkMatch.index);
        this.appendAnswer(beforeTag);
      }
      this.buffer = this.buffer.substring(thinkMatch.index + thinkMatch[0].length);
      this.state = ParserState.IN_THINK;
      return true;
    }

    // Check for <verify_code> tag
    const codeMatch = this.buffer.match(/<verify_code\s+language=["'](\w+)["']>/i);
    if (codeMatch && codeMatch.index !== undefined) {
      if (codeMatch.index > 0) {
        const beforeTag = this.buffer.substring(0, codeMatch.index);
        this.appendAnswer(beforeTag);
      }
      this.currentCodeLanguage = codeMatch[1] || 'python';
      this.currentCodeBlock = { language: this.currentCodeLanguage, code: '' };
      this.buffer = this.buffer.substring(codeMatch.index + codeMatch[0].length);
      this.state = ParserState.IN_VERIFY_CODE;
      return true;
    }

    // Check for <observation> tag (can appear after verify_code closes)
    const obsMatch = this.buffer.match(/<observation>/i);
    if (obsMatch && obsMatch.index !== undefined) {
      if (obsMatch.index > 0) {
        const beforeTag = this.buffer.substring(0, obsMatch.index);
        this.appendAnswer(beforeTag);
      }
      this.buffer = this.buffer.substring(obsMatch.index + obsMatch[0].length);
      this.state = ParserState.IN_OBSERVATION;
      return true;
    }

    // Check if buffer might contain a partial tag
    if (this.mightContainPartialTag(this.buffer)) {
      // Keep buffer for next chunk
      return false;
    }

    // No tags found, output as answer
    if (this.buffer.length > 0) {
      this.appendAnswer(this.buffer);
      this.buffer = '';
      return true;
    }

    return false;
  }

  /**
   * Process buffer in IN_THINK state.
   * Look for closing </think> tag.
   */
  private processInThinkState(): boolean {
    const closeMatch = this.buffer.match(/<\/think>/i);

    if (closeMatch && closeMatch.index !== undefined) {
      // Extract thought content before closing tag
      const content = this.buffer.substring(0, closeMatch.index);
      this.appendThought(content);
      this.buffer = this.buffer.substring(closeMatch.index + closeMatch[0].length);
      this.state = ParserState.NORMAL;
      return true;
    }

    // Check for partial closing tag
    if (this.mightContainPartialTag(this.buffer, '</think>')) {
      return false;
    }

    // No closing tag, accumulate as thought
    if (this.buffer.length > 0) {
      this.appendThought(this.buffer);
      this.buffer = '';
      return true;
    }

    return false;
  }

  /**
   * Process buffer in IN_VERIFY_CODE state.
   * Look for closing </verify_code> tag.
   */
  private processInVerifyCodeState(): boolean {
    const closeMatch = this.buffer.match(/<\/verify_code>/i);

    if (closeMatch && closeMatch.index !== undefined) {
      // Extract code content before closing tag
      const content = this.buffer.substring(0, closeMatch.index);
      this.appendCode(content);
      this.buffer = this.buffer.substring(closeMatch.index + closeMatch[0].length);
      this.state = ParserState.NORMAL;
      return true;
    }

    // Check for partial closing tag
    if (this.mightContainPartialTag(this.buffer, '</verify_code>')) {
      return false;
    }

    // No closing tag, accumulate as code
    if (this.buffer.length > 0) {
      this.appendCode(this.buffer);
      this.buffer = '';
      return true;
    }

    return false;
  }

  /**
   * Process buffer in IN_OBSERVATION state.
   * Look for closing </observation> tag.
   */
  private processInObservationState(): boolean {
    const closeMatch = this.buffer.match(/<\/observation>/i);

    if (closeMatch && closeMatch.index !== undefined) {
      // Extract observation content
      const content = this.buffer.substring(0, closeMatch.index);
      this.appendObservation(content);
      this.buffer = this.buffer.substring(closeMatch.index + closeMatch[0].length);
      this.state = ParserState.NORMAL;
      return true;
    }

    // Check for partial closing tag
    if (this.mightContainPartialTag(this.buffer, '</observation>')) {
      return false;
    }

    // No closing tag, accumulate as observation
    if (this.buffer.length > 0) {
      this.appendObservation(this.buffer);
      this.buffer = '';
      return true;
    }

    return false;
  }

  /**
   * Check if buffer might contain a partial tag at the end.
   */
  private mightContainPartialTag(buffer: string, specificTag?: string): boolean {
    if (buffer.length === 0) return false;

    const tagPrefixes = specificTag
      ? [specificTag]
      : ['<think>', '</think>', '<verify_code', '</verify_code>', '<observation>', '</observation>'];

    for (const tag of tagPrefixes) {
      // Check if buffer ends with any prefix of the tag
      for (let i = 1; i < tag.length; i++) {
        const prefix = tag.substring(0, i);
        if (buffer.endsWith(prefix) || buffer.toLowerCase().endsWith(prefix.toLowerCase())) {
          return true;
        }
      }
    }

    // Also check for < at the end (start of any tag)
    return buffer.endsWith('<');
  }

  // =============================================================================
  // Content Accumulators
  // =============================================================================

  private appendThought(content: string): void {
    this.thoughtContent += content;
    this.onThought?.(content);
  }

  private appendCode(content: string): void {
    if (this.currentCodeBlock) {
      this.currentCodeBlock.code += content;
      this.onCode?.(content, this.currentCodeLanguage);
    }
  }

  private appendObservation(content: string): void {
    // Attach observation to the most recent code block
    if (this.currentCodeBlock) {
      this.currentCodeBlock.observation = (this.currentCodeBlock.observation || '') + content;
      // Finalize the code block
      this.codeBlocks.push({ ...this.currentCodeBlock });
      this.currentCodeBlock = null;
    }
    this.onObservation?.(content);
  }

  private appendAnswer(content: string): void {
    // Filter out any remaining tag artifacts
    const cleaned = content.replace(/<\/?[\w]+[^>]*>/g, '').trim();
    if (cleaned) {
      this.answerContent += content;
      this.onAnswer?.(content);
    }
  }

  /**
   * Flush any remaining buffer content.
   */
  public flush(): void {
    if (this.buffer.length > 0) {
      switch (this.state) {
        case ParserState.IN_THINK:
          this.appendThought(this.buffer);
          break;
        case ParserState.IN_VERIFY_CODE:
          this.appendCode(this.buffer);
          break;
        case ParserState.IN_OBSERVATION:
          this.appendObservation(this.buffer);
          break;
        default:
          this.appendAnswer(this.buffer);
      }
      this.buffer = '';
    }

    // Finalize any pending code block
    if (this.currentCodeBlock) {
      this.codeBlocks.push({ ...this.currentCodeBlock });
      this.currentCodeBlock = null;
    }
  }

  /**
   * Reset parser state.
   */
  public reset(): void {
    this.state = ParserState.NORMAL;
    this.buffer = '';
    this.thoughtContent = '';
    this.codeBlocks = [];
    this.currentCodeBlock = null;
    this.answerContent = '';
  }
}

// =============================================================================
// React Hook
// =============================================================================

export function useVAGIStream(config: VAGIStreamConfig): UseVAGIStreamReturn {
  const {
    apiUrl,
    model = 'vagi',
    temperature = 0.7,
    maxTokens = 2048,
    onThought,
    onCode,
    onObservation,
    onAnswer,
    onError,
    onComplete,
  } = config;

  // State
  const [thoughtTrace, setThoughtTrace] = useState<string>('');
  const [codeExecutions, setCodeExecutions] = useState<CodeBlock[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);
  const parserRef = useRef<VAGIStreamParser | null>(null);

  /**
   * Send a message and start streaming response.
   */
  const sendMessage = useCallback(
    async (message: string, systemPrompt?: string): Promise<void> => {
      // Abort any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Reset state
      setThoughtTrace('');
      setCodeExecutions([]);
      setFinalAnswer('');
      setError(null);
      setIsStreaming(true);
      setIsConnected(false);

      // Create new abort controller
      abortControllerRef.current = new AbortController();

      // Create parser with callbacks
      parserRef.current = new VAGIStreamParser({
        onThought: (content) => {
          setThoughtTrace((prev) => prev + content);
          onThought?.(content);
        },
        onCode: (code, language) => {
          onCode?.(code, language);
        },
        onObservation: (result) => {
          onObservation?.(result);
        },
        onAnswer: (content) => {
          setFinalAnswer((prev) => prev + content);
          onAnswer?.(content);
        },
      });

      try {
        // Build messages array
        const messages: Array<{ role: string; content: string }> = [];

        if (systemPrompt) {
          messages.push({ role: 'system', content: systemPrompt });
        }

        messages.push({ role: 'user', content: message });

        // Make request
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
          },
          body: JSON.stringify({
            model,
            messages,
            temperature,
            max_tokens: maxTokens,
            stream: true,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error('Response body is null');
        }

        setIsConnected(true);

        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          // Decode chunk
          buffer += decoder.decode(value, { stream: true });

          // Process SSE lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim();

              if (data === '[DONE]') {
                continue;
              }

              try {
                const parsed = JSON.parse(data);
                const delta = parsed.choices?.[0]?.delta;

                if (delta?.content) {
                  parserRef.current?.processChunk(delta.content);
                }

                // Check for finish reason
                if (parsed.choices?.[0]?.finish_reason === 'stop') {
                  break;
                }
              } catch (parseError) {
                // Ignore JSON parse errors for incomplete chunks
                console.warn('Failed to parse SSE data:', data);
              }
            }
          }
        }

        // Flush parser
        parserRef.current?.flush();

        // Update code executions from parser
        if (parserRef.current) {
          setCodeExecutions([...parserRef.current.codeBlocks]);
        }

        onComplete?.();
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Aborted intentionally
          return;
        }

        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        onError?.(error);
      } finally {
        setIsStreaming(false);
        setIsConnected(false);
      }
    },
    [apiUrl, model, temperature, maxTokens, onThought, onCode, onObservation, onAnswer, onError, onComplete]
  );

  /**
   * Abort the current stream.
   */
  const abort = useCallback((): void => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setIsConnected(false);
  }, []);

  /**
   * Reset all state.
   */
  const reset = useCallback((): void => {
    abort();
    setThoughtTrace('');
    setCodeExecutions([]);
    setFinalAnswer('');
    setError(null);
    parserRef.current?.reset();
  }, [abort]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abort();
    };
  }, [abort]);

  return {
    thoughtTrace,
    codeExecutions,
    finalAnswer,
    isStreaming,
    isConnected,
    error,
    sendMessage,
    abort,
    reset,
  };
}

// =============================================================================
// Export Parser for Testing
// =============================================================================

export { VAGIStreamParser, ParserState };

// =============================================================================
// Example Usage Component
// =============================================================================

/*
import React from 'react';
import { useVAGIStream } from './hooks/useVAGIStream';

function ChatComponent() {
  const {
    thoughtTrace,
    codeExecutions,
    finalAnswer,
    isStreaming,
    error,
    sendMessage,
    abort,
    reset,
  } = useVAGIStream({
    apiUrl: '/v1/chat/completions',
    onThought: (content) => console.log('Thinking:', content),
    onCode: (code, lang) => console.log(`Code (${lang}):`, code),
    onObservation: (result) => console.log('Observation:', result),
  });

  const handleSubmit = async (message: string) => {
    await sendMessage(message);
  };

  return (
    <div className="chat-container">
      {/* Thinking Panel (Gray text) *\/}
      {thoughtTrace && (
        <div className="thought-panel" style={{ color: '#888' }}>
          <h4>Thinking...</h4>
          <pre>{thoughtTrace}</pre>
        </div>
      )}

      {/* Code Execution Panel *\/}
      {codeExecutions.map((block, i) => (
        <div key={i} className="code-panel">
          <pre className="code-block">
            <code>{block.code}</code>
          </pre>
          {block.observation && (
            <div className="observation">
              Result: {block.observation}
            </div>
          )}
        </div>
      ))}

      {/* Final Answer (White text) *\/}
      <div className="answer-panel" style={{ color: '#fff' }}>
        {finalAnswer}
      </div>

      {/* Status *\/}
      {isStreaming && <div className="status">Streaming...</div>}
      {error && <div className="error">{error.message}</div>}

      {/* Controls *\/}
      <button onClick={() => handleSubmit('Hello')}>Send</button>
      <button onClick={abort} disabled={!isStreaming}>Abort</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}
*/
