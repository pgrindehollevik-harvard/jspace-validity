"""Artifact identities used to stage Qwen3 models for the scaling extension.

These profiles deliberately contain *only* immutable artifact locations and model
shapes.  They do not choose an ablation band or analysis rule for a new model: those
belong in the extension's pre-registration, not in deployment plumbing.
"""

from dataclasses import dataclass

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "0731326edff4ae730ffc5356fe1a4728c748b3a6"
COMPANION_REPO = "https://github.com/anthropics/jacobian-lens.git"
COMPANION_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


@dataclass(frozen=True)
class Qwen3Profile:
    key: str
    model_id: str
    lens_file: str
    n_layers: int
    hidden_size: int
    model_revision: str


QWEN3_PROFILES: dict[str, Qwen3Profile] = {
    "qwen3-8b": Qwen3Profile(
        "qwen3-8b",
        "Qwen/Qwen3-8B",
        "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        36,
        4096,
        "b968826d9c46dd6066d109eabc6255188de91218",
    ),
    "qwen3-14b": Qwen3Profile(
        "qwen3-14b",
        "Qwen/Qwen3-14B",
        "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
        40,
        5120,
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
    ),
    # Version-3 registration (extension/PREREGISTRATION_V3.md): the two follow-up
    # scales that the version-2 preregistration reserved for separate recording.
    "qwen3-1.7b": Qwen3Profile(
        "qwen3-1.7b",
        "Qwen/Qwen3-1.7B",
        "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        28,
        2048,
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    ),
    "qwen3-32b": Qwen3Profile(
        "qwen3-32b",
        "Qwen/Qwen3-32B",
        "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt",
        64,
        5120,
        "9216db5781bf21249d130ec9da846c4624c16137",
    ),
}


def get_profile(key: str) -> Qwen3Profile:
    try:
        return QWEN3_PROFILES[key.lower()]
    except KeyError as exc:
        choices = ", ".join(QWEN3_PROFILES)
        raise ValueError(f"unknown model profile {key!r}; choose one of: {choices}") from exc
