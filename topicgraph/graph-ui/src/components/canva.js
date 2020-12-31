import React from "react";
import { ForceGraph2D, ForceGraph3D } from "react-force-graph";
import SpriteText from "three-spritetext";
export default function Canva({ graph, is3d }) {
  const data = require("../data/" + graph);
  return (
    <div>
      {/* <ExpandableGraph graphData={init_data} />*/}

      {is3d ? (
        <ForceGraph3D
          backgroundColor={"#000"}
          graphData={data}
          nodeLabel="id"
          linkAutoColorBy="weight"
          linkOpacity={0.5}
          linkWidth={1}
          linkDirectionalParticles={1}
          nodeThreeObject={(node) => {
            const sprite = new SpriteText(node.id);
            // sprite.color = node.color;
            sprite.textHeight = 8;
            return sprite;
          }}
        />
      ) : (
        <ForceGraph2D
          backgroundColor={"#000"}
          graphData={data}
          nodeLabel="id"
          linkAutoColorBy="weight"
          // linkWidth="weight"
          linkDirectionalParticles={1}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.id;
            const fontSize = 15 / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            const textWidth = ctx.measureText(label).width;
            const bckgDimensions = [textWidth, fontSize].map(
              (n) => n + fontSize * 0.2
            ); // some padding

            ctx.fillStyle = "#000";
            ctx.fillRect(
              node.x - bckgDimensions[0] / 2,
              node.y - bckgDimensions[1] / 2,
              ...bckgDimensions
            );

            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#fff";
            ctx.fillText(label, node.x, node.y);
          }}
        />
      )}
    </div>
  );
}
