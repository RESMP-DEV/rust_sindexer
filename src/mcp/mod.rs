pub mod hybrid;
pub mod indexer;
pub mod manifest;
pub mod state;

#[cfg(feature = "mcp")]
pub mod tools;

pub use hybrid::HybridHit;
pub use state::{
    create_default_shared_state, create_shared_state, create_shared_state_with_components,
    ClearResult, ContextState, IndexResult, SearchResult, SharedState,
};

#[cfg(feature = "mcp")]
pub use tools::CodebaseTools;
