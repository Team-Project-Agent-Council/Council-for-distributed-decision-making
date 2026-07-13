import asyncio
import click
from dotenv import load_dotenv
load_dotenv()
from council.graph import build_graph


async def run(image_path: str) -> None:
    graph = build_graph()
    async for chunk in graph.astream(
        {"image_path": image_path},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            print(f"\n[{node}]")
            for key, value in update.items():
                if value:
                    print(f"{key}: {value}")


@click.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Path to the image file (jpg, png, webp).")
def main(image: str) -> None:
    asyncio.run(run(image))

if __name__ == "__main__":
    main()
