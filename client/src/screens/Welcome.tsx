import { ConnectButton } from "@pipecat-ai/voice-ui-kit";

interface WelcomeProps {
  onConnect?: () => void | Promise<void>;
  onDisconnect?: () => void | Promise<void>;
}

// Phrasings that exercise different parts of the app so the user sees
// the voice agent driving the UI (navigate), the catalog (play), and
// the multi-source Related-tab fan-out (similar artists).
const EXAMPLES = [
  "Show me Radiohead",
  "Play OK Computer",
  "Show related artists",
];

export function Welcome({ onConnect, onDisconnect }: WelcomeProps) {
  return (
    <div className="welcome">
      <div className="welcome-card">
        <div className="welcome-eyebrow">Voice Music Player</div>
        <h1 className="welcome-title">Browse music with your voice</h1>
        <div className="welcome-examples">
          <div className="welcome-examples-label">Try saying</div>
          <ul className="welcome-examples-list">
            {EXAMPLES.map((phrase) => (
              <li key={phrase} className="welcome-examples-item">
                <span className="welcome-examples-quote">“{phrase}”</span>
              </li>
            ))}
          </ul>
        </div>
        <ConnectButton
          size="lg"
          onConnect={onConnect}
          onDisconnect={onDisconnect}
        />
      </div>
    </div>
  );
}
