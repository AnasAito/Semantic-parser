import React, { useState } from "react";
import Canva from "./components/canva";
import Dir from "./components/dir";

const dirs = require("./data/index.json");
export default function App() {
  const [graph, setGraph] = useState(dirs.graphs[0]);
  const [on, setOn] = useState(false);
  return (
    <div className="  flex  flex-col ">
      <div className="flex flex-row justify-between">
        <div className=" bg-indigo-200 rounded m-3 w-1/3 p-3 ">
          <Dir
            setGraph={setGraph}
            itemOnClick={(item) => setGraph(item)}
            items={dirs.graphs}
            graph={graph}
          />
        </div>
        <div className="flex  m-3 items-center">
          <div class="flex items-center">
            <button
              onClick={() => {
                setOn(!on);
              }}
              type="button"
              aria-pressed="false"
              aria-labelledby="toggleLabel"
              class={`${
                on ? "bg-indigo-600" : "bg-gray-200"
              } relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
            >
              <span class="sr-only">Use setting</span>

              <span
                aria-hidden="true"
                class={`${
                  on ? "translate-x-5" : "translate-x-0"
                } inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200`}
              ></span>
            </button>
            <span class="ml-3" id="toggleLabel">
              <span class="text-sm font-medium text-gray-900">3d mode</span>
            </span>
          </div>
        </div>
      </div>
      <div className="">
        {" "}
        <Canva graph={graph} is3d={on} />
      </div>
    </div>
  );
}
