import React from "react";
import { ForceGraph2D, ForceGraph3D } from "react-force-graph";
import SpriteText from "three-spritetext";
export default function Canva({ graph, is3d }) {
  const data = require("../data/" + graph);

  const graph3dAttr = {
    backgroundColor: "#000",
    graphData: data,
    nodeLabel: "id",
    // nodeColor: "community",

    linkAutoColorBy: "weight",
    linkOpacity: 0.5,
    linkWidth: 1,
    linkDirectionalParticles: 1,
    nodeValue: "degree",
  };
  const detailTresh = 0;
  const graph2dAttr = {
    backgroundColor: "#000",

    graphData: data,
    nodeLabel: "id",
    //linkAutoColorBy: "weight",

    linkColor: (edge) => {
      let c = edge["new"] == 1 ? "red" : "#000";

      return c;
    },
    // linkWidth="weight"
    // linkCurvature={0.1}

    // linkDirectionalParticles: 1,
    nodeCanvasObject: (node, ctx, globalScale) => {
      // text
      if (node.degree > detailTresh) {
        const color_dict = {
          cellulose: "#228B22",
          plant: "#7FFF00",
          sludge: "#8A2BE2",
          waste: "#6027E9",
          material: "#FFD700",
          phosphate: "#A9A9A9",
          soil: "#DEB887",
        };

        const label = node.id;
        const color = color_dict[node.group];
        //  const group = parseInt(node.group) == -1 ? 8 : parseInt(node.group);
        const val = parseFloat(node.score) * 300;
        //let value = degree;
        // draw cirle
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(node.x, node.y, val, 0, 2 * Math.PI, false);
        // text

        ctx.fill();
        ctx.fillStyle = color;
        ctx.font = `${val * 3}px serif`;
        ctx.fillText(label, node.x + val + 5, node.y + val / 2);
      }
    },
  };

  return (
    <div>
      {/* <ExpandableGraph graphData={init_data} />*/}

      {is3d ? (
        <ForceGraph3D {...graph3dAttr} />
      ) : (
        <ForceGraph2D {...graph2dAttr} />
      )}
    </div>
  );
}
