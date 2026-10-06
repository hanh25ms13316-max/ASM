import asyncio, pathlib, json
from playwright.async_api import async_playwright
MER = pathlib.Path("/tmp/claude-0/-home-claude/18a594be-8194-5506-8022-55d4b9bf1f05/scratchpad/pdfbuild/node_modules/mermaid/dist/mermaid.min.js").read_text()
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page()
        await pg.set_content("<html><body></body></html>")
        await pg.add_script_tag(content=MER)
        await pg.evaluate("mermaid.initialize({startOnLoad:false, theme:'neutral', fontFamily:'Arial, sans-serif', themeVariables:{fontSize:'15px'}})")
        for name in ["pipeline","scoring","training"]:
            src = pathlib.Path(f"{name}.mmd").read_text()
            svg = await pg.evaluate("async ([id,src]) => (await mermaid.render(id, src)).svg", [name+"svg", src])
            pathlib.Path(f"{name}.svg").write_text(svg); print(name, len(svg))
        await b.close()
asyncio.run(main())
