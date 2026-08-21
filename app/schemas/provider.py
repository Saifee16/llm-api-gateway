from pydantic import BaseModel


class ProviderData(BaseModel):
    name: str
    configured: bool
    capabilities: list[str]


class ProvidersData(BaseModel):
    providers: list[ProviderData]


class ModelTargetData(BaseModel):
    provider: str
    model: str
    configured: bool


class ModelAliasData(BaseModel):
    alias: str
    targets: list[ModelTargetData]


class ModelsData(BaseModel):
    aliases: list[ModelAliasData]
