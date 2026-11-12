from pathlib import Path
from typing import Optional, Union, Dict, Any

from spectra.hardware import HardwareInspector, HardwareProfile
from spectra.compressor import SpectraCompressor
from spectra.engine import SpectraUniversalEngine

class Spectra:
    @classmethod
    def probe_hardware(cls, device_index: int = 0) -> HardwareProfile:
        return HardwareInspector.probe(device_index=device_index)

    @classmethod
    def compress(
        cls,
        model_id_or_path: Union[str, Path],
        output_container: Optional[Union[str, Path]] = None,
        group_size: int = 32,
        outlier_ratio: float = 0.01,
        trust_remote_code: bool = False
    ) -> Dict[str, Any]:
        model_path = Path(model_id_or_path)
        if output_container is None:
            output_container = model_path.parent / f"{model_path.name}_spectra.trdc"

        compressor = SpectraCompressor(
            group_size=group_size,
            base_outlier_ratio=outlier_ratio,
            adaptive_kurtosis=True,
            trust_remote_code=trust_remote_code
        )
        return compressor.compress(model_id_or_path, output_container)

    @classmethod
    def serve(
        cls,
        model_or_container: Union[str, Path],
        model_dir: Optional[Union[str, Path]] = None,
        device_index: int = 0,
        auto_compress: bool = True,
        trust_remote_code: bool = False
    ) -> SpectraUniversalEngine:
        target_path = Path(model_or_container).resolve()

        if target_path.suffix == ".trdc" and target_path.exists():
            return SpectraUniversalEngine.load(
                trdc_path=target_path,
                model_dir=model_dir,
                device_index=device_index,
                trust_remote_code=trust_remote_code
            )

        if target_path.is_dir():
            local_trdc = list(target_path.glob("model.trdc")) + list(target_path.glob("*.trdc"))
            if local_trdc:
                chosen_trdc = local_trdc[0]
                return SpectraUniversalEngine.load(
                    trdc_path=chosen_trdc,
                    model_dir=target_path,
                    device_index=device_index,
                    trust_remote_code=trust_remote_code
                )

        default_container = target_path.parent / "compressed_models" / f"{target_path.stem}_trdc.trdc"
        if not default_container.exists():
            default_container = target_path.parent / f"{target_path.stem}_spectra.trdc"
        if not default_container.exists() and (target_path.parent / "compressed_models" / f"{target_path.stem}_spectra.trdc").exists():
            default_container = target_path.parent / "compressed_models" / f"{target_path.stem}_spectra.trdc"

        if default_container.exists():
            return SpectraUniversalEngine.load(
                trdc_path=default_container,
                model_dir=target_path,
                device_index=device_index,
                trust_remote_code=trust_remote_code
            )

        if auto_compress:
            out_trdc = target_path / "model.trdc" if target_path.is_dir() else default_container
            cls.compress(target_path, out_trdc, trust_remote_code=trust_remote_code)
            return SpectraUniversalEngine.load(
                trdc_path=out_trdc,
                model_dir=target_path,
                device_index=device_index,
                trust_remote_code=trust_remote_code
            )
        else:
            raise FileNotFoundError(f"Container not found and auto_compress is False for: {target_path}")

    load = serve

