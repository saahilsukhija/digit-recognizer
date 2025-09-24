import { useState } from "react";
import "../style/Output.css";

function Output(props) {
  return (
    <div className="output-board">
      <h1>{props.number}</h1>
    </div>
  );
}

export default Output;
