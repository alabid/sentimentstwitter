/*
 * main.js ->
 *  Contains code to draw donut chart for 
 *  sentiment analysis results.
 */

// sent-0 -> #FFC2C2; (negative)
// sent-1 -> #53DB21; (positive)
var colors = ["#FFC2C2", "#53DB21"];
/*
 * Draw donut chart here
 */
function drawVisuals() {
    var elem = $("#visuals");
    var pos = parseInt(elem.attr("pos"));
    var neg = parseInt(elem.attr("neg"));

    var dataset = {
        sentiments: [neg, pos]
    };

    var width = 560,
        height = 400,
        radius = Math.min(width, height) / 2;
        
    var pie = d3.layout.pie()
            .sort(null);
    
    var arc = d3.svg.arc()
            .innerRadius(radius - 100)
            .outerRadius(radius - 5);
    
    var svg = d3.select($("#visuals div")[0]).append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform"
                  , "translate(" + width / 2 + "," + height / 2 + ")");
    
    var path = svg.selectAll("path")
            .data(pie(dataset.sentiments))
            .enter().append("path")
            .attr("fill", function(d, i) { 
                return colors[i];
            })
            .attr("d", arc);
}

$( document ).ready(function() {
    $("span.timeago").timeago();

    if ($("input[name=query]").attr("value").trim().length > 0) {
        $("#tweets").show();

        drawVisuals();

        $("#visuals").show();
    }

    if (document.URL.indexOf("classifier-type=1") != -1) {
        $("option[value='1']").attr("selected", "selected");
    } else {
        $("option[value='0']").attr("selected", "selected");
    }    
});
