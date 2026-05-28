import type {
  Album,
  Artist as ArtistType,
  ArtistTab,
  DiscoverySection,
  MinimalArtist,
  Song,
} from "../types";
import { Grid, GridCell } from "../components/Grid";

interface ArtistProps {
  artist: ArtistType;
  activeTab: ArtistTab;
  onSelectItem: (kind: "album" | "song", item: Album | Song) => void;
  onSelectRelated: (artist: MinimalArtist) => void;
  onSelectTab: (tab: ArtistTab) => void;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

const TABS: { key: ArtistTab; label: string }[] = [
  { key: "albums", label: "Albums" },
  { key: "songs", label: "Songs" },
  { key: "related", label: "Related artists" },
];

export function Artist({
  artist,
  activeTab,
  onSelectItem,
  onSelectRelated,
  onSelectTab,
}: ArtistProps) {
  const columns = 8;
  const albumCoverById = new Map(artist.albums.map((a) => [a.id, a.cover_url]));
  const relatedSections = artist.related_sections ?? null;
  return (
    <div className="screen artist-screen">
      <div className="artist-header">
        <img src={artist.image_url} alt="" className="artist-header-image" />
        <div className="artist-header-text">
          <h1 className="screen-title">{artist.name}</h1>
          {artist.genre && <div className="artist-genre">{artist.genre}</div>}
          {artist.short_description && (
            <p className="artist-short-description">
              {artist.short_description}
            </p>
          )}
        </div>
      </div>
      <div className="tab-bar" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.key}
            className={`tab-button ${activeTab === tab.key ? "active" : ""}`}
            onClick={() => onSelectTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {activeTab === "albums" && (
        <Grid columns={columns}>
          {artist.albums.map((album, index) => (
            <GridCell
              key={album.id}
              title={album.title}
              subtitle={String(album.year)}
              imageUrl={album.cover_url}
              row={Math.floor(index / columns) + 1}
              col={(index % columns) + 1}
              onClick={() => onSelectItem("album", album)}
            />
          ))}
        </Grid>
      )}
      {activeTab === "songs" && (
        <Grid columns={columns}>
          {artist.songs.map((song, index) => (
            <GridCell
              key={song.id}
              title={song.title}
              subtitle={formatDuration(song.duration_seconds)}
              imageUrl={
                song.cover_url ?? albumCoverById.get(song.album_id) ?? artist.image_url
              }
              row={Math.floor(index / columns) + 1}
              col={(index % columns) + 1}
              onClick={() => onSelectItem("song", song)}
            />
          ))}
        </Grid>
      )}
      {activeTab === "related" && (
        <RelatedTab
          sections={relatedSections}
          columns={columns}
          onSelect={onSelectRelated}
        />
      )}
    </div>
  );
}

interface RelatedTabProps {
  sections: DiscoverySection[] | null;
  columns: number;
  onSelect: (artist: MinimalArtist) => void;
}

// Placeholder cell count while a worker is still running. Mirrors the
// shape of the eventual grid so the layout doesn't jump when results
// land.
const SKELETON_CELLS = 6;

const STATUS_LABEL: Record<DiscoverySection["status"], string> = {
  running: "searching",
  completed: "done",
  cancelled: "cancelled",
  error: "error",
};

function RelatedTab({ sections, columns, onSelect }: RelatedTabProps) {
  if (sections === null) {
    // Server hasn't kicked off discovery yet (this is the brief window
    // between the tab click and the re-emit with the empty sections).
    return <div className="tab-loading">Loading similar artists…</div>;
  }
  return (
    <div className="related-tab">
      {sections.map((section) => (
        <RelatedSection
          key={section.worker_name}
          section={section}
          columns={columns}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

interface RelatedSectionProps {
  section: DiscoverySection;
  columns: number;
  onSelect: (artist: MinimalArtist) => void;
}

function RelatedSection({ section, columns, onSelect }: RelatedSectionProps) {
  const isRunning = section.status === "running";
  const showEmpty = !isRunning && section.artists.length === 0;
  return (
    <section className={`discovery-section discovery-section--${section.status}`}>
      <div className="discovery-section-header">
        <h2 className="grid-label">{section.label}</h2>
        <span className={`discovery-section-status discovery-section-status--${section.status}`}>
          {STATUS_LABEL[section.status]}
        </span>
      </div>
      {isRunning && section.artists.length === 0 && (
        <Grid columns={columns}>
          {Array.from({ length: SKELETON_CELLS }).map((_, i) => (
            <div key={i} className="discovery-skeleton" aria-hidden="true" />
          ))}
        </Grid>
      )}
      {section.artists.length > 0 && (
        <Grid columns={columns}>
          {section.artists.map((a, index) => {
            const row = Math.floor(index / columns) + 1;
            const col = (index % columns) + 1;
            // LLM-only rows have no id and no image — render as a
            // non-clickable label cell.
            if (!a.id) {
              return (
                <div
                  key={`${section.worker_name}:${index}:${a.name}`}
                  className="grid-cell grid-cell--inert"
                  data-row={row}
                  data-col={col}
                >
                  <div className="grid-cell-body">
                    <div className="grid-cell-title">{a.name}</div>
                    <div className="grid-cell-subtitle">Suggested</div>
                  </div>
                </div>
              );
            }
            return (
              <GridCell
                key={`${section.worker_name}:${a.id}`}
                title={a.name}
                subtitle="Artist"
                imageUrl={a.image_url}
                row={row}
                col={col}
                onClick={() => onSelect(a)}
              />
            );
          })}
        </Grid>
      )}
      {showEmpty && <div className="discovery-empty">No results from this source.</div>}
    </section>
  );
}
