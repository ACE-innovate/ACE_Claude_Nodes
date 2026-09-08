import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// "choose file to upload" button on ACE_Claude_Push_File.
// No file-type filter. The file is POSTed to the node's own
// /ace_claude/upload endpoint, held in server RAM only (never written to
// disk), and pushed to the Anthropic Files API when the node runs.

app.registerExtension({
  name: "ACE.Claude.PushFileUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ACE_Claude_Push_File") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      const fileWidget = this.widgets.find((w) => w.name === "file");
      if (!fileWidget) return r;

      this.addWidget(
        "button",
        "upload",
        "choose file to upload",
        () => {
          const input = document.createElement("input");
          input.type = "file"; // no accept attribute -> ALL file types
          input.onchange = async () => {
            if (!input.files || !input.files.length) return;
            const file = input.files[0];
            const body = new FormData();
            body.append("file", file);
            let resp;
            try {
              resp = await api.fetchApi("/ace_claude/upload", {
                method: "POST",
                body,
              });
            } catch (e) {
              alert("Upload failed: " + e);
              return;
            }
            if (resp.status !== 200) {
              alert("Upload failed: HTTP " + resp.status + " " + (await resp.text()));
              return;
            }
            const name = file.name;
            if (
              Array.isArray(fileWidget.options.values) &&
              !fileWidget.options.values.includes(name)
            ) {
              fileWidget.options.values.push(name);
            }
            fileWidget.value = name;
            if (fileWidget.callback) fileWidget.callback(name);
            app.graph.setDirtyCanvas(true);
          };
          input.click();
        },
        { serialize: false }
      );

      return r;
    };
  },
});
