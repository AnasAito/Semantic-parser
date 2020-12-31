import React, { useState } from "react";
import OutsideAlerter from "./OutsideClicker";
export default function Dir({
  items,
  itemOnClick,
  graph,
  title = "choose a graph",
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="flex flex-col">
      {title ? (
        <label class=" text-base pb-2 font-semibold  text-black ">
          {title}
        </label>
      ) : (
        <></>
      )}
      <OutsideAlerter onOutside={() => setShow(false)}>
        <div className={`relative    `}>
          <span className="rounded-md shadow-sm">
            <button
              id="sort-menu"
              type="button"
              className="inline-flex  justify-between w-full rounded-md border border-gray-300 px-4 py-2 bg-white text-sm leading-5 font-medium text-gray-700 hover:text-gray-500 focus:outline-none focus:border-blue-300 focus:shadow-outline-blue active:bg-gray-50 active:text-gray-800 transition ease-in-out duration-150"
              aria-haspopup="true"
              aria-expanded="false"
              onClick={() => setShow(!show)}
            >
              {graph}
              <svg
                className="ml-2.5 -mr-1.5 h-5 w-5 text-gray-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </span>
          <div
            className={`transform transition ease-in-out duration-500 sm:duration-700  ${
              show ? "opacity-100" : " opacity-0 hidden"
            } origin-top-right z-10 absolute left-0 mt-2  rounded-md shadow-lg`}
          >
            <div className="rounded-md bg-white shadow-xs  h-96  overflow-y-scroll ">
              <div
                className="py-1"
                role="menu"
                aria-orientation="vertical"
                aria-labelledby="sort-menu"
              >
                {items.map((item) => (
                  <button
                    key={item}
                    className=" px-4 py-2 flex justify-start w-full text-sm leading-5 text-gray-700 hover:bg-gray-100 hover:text-gray-900 focus:outline-none focus:bg-gray-100 focus:text-gray-900"
                    role="menuitem"
                    onClick={() => {
                      itemOnClick(item);
                      setShow(!show);
                    }}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </OutsideAlerter>
    </div>
  );
}
