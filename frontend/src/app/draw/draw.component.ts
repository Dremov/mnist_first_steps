import { Component, OnInit, Input, ViewChild, ElementRef, Output } from '@angular/core';
import { Observable } from 'rxjs/Observable';
import 'rxjs/add/observable/fromEvent';
import 'rxjs/add/operator/switchMap';
import 'rxjs/add/operator/takeUntil';
import 'rxjs/add/operator/pairwise';


@Component({
  selector: 'app-draw',
  templateUrl: './draw.component.html',
  styleUrls: ['./draw.component.scss']
})
export class DrawComponent implements OnInit {

	@ViewChild('canvas') public canvas: ElementRef;


	// setting a width and height for the canvas
	@Input() public width = 300;
	@Input() public height = 300;

	private cx: CanvasRenderingContext2D;

  constructor() { }

  ngOnInit() {
		// get the context
    const canvasEl: HTMLCanvasElement = this.canvas.nativeElement;
    this.cx = canvasEl.getContext('2d');

    // set the width and height
    canvasEl.width = this.width;
    canvasEl.height = this.height;

    // set some default properties about the line
    this.cx.lineWidth = 20;
    this.cx.lineCap = 'round';
		this.cx.strokeStyle = '#FFF';

    // we'll implement this method to start capturing mouse events
    this.captureEvents(canvasEl);
	}

	clearCanvas() {
		this.cx.clearRect(0, 0, this.cx.canvas.width, this.cx.canvas.height); // Clears the canvas
	}

	private captureEvents(canvasEl: HTMLCanvasElement) {
		Observable
			// this will capture all mousedown events from teh canvas element
			.fromEvent(canvasEl, 'mousedown')
			.switchMap((e) => {
				return Observable
					// after a mouse down, we'll record all mouse moves
					.fromEvent(canvasEl, 'mousemove')
					// we'll stop (and unsubscribe) once the user releases the mouse
					// this will trigger a mouseUp event
					.takeUntil(Observable.fromEvent(canvasEl, 'mouseup'))
					// pairwise lets us get the previous value to draw a line from
					// the previous point to the current point
					.pairwise()
			})
			.subscribe((res: [MouseEvent, MouseEvent]) => {
				const rect = canvasEl.getBoundingClientRect();

				// previous and current position with the offset
				const prevPos = {
					x: res[0].clientX - rect.left,
					y: res[0].clientY - rect.top
				};

				const currentPos = {
					x: res[1].clientX - rect.left,
					y: res[1].clientY - rect.top
				};

				// this method we'll implement soon to do the actual drawing
				this.drawOnCanvas(prevPos, currentPos);
			});
	}

	private drawOnCanvas(
		prevPos: { x: number, y: number },
		currentPos: { x: number, y: number }
	) {
		// incase the context is not set
		if (!this.cx) { return; }

		// start our drawing path
		this.cx.beginPath();

		// we're drawing lines so we need a previous position
		if (prevPos) {
			// sets the start point
			this.cx.moveTo(prevPos.x, prevPos.y); // from
			// draws a line from the start pos until the current position
			this.cx.lineTo(currentPos.x, currentPos.y);

			// strokes the current path with the styles we set earlier
			this.cx.stroke();
		}
	}

	public getSavedImage(): string {
		var dataURL = this.cx.canvas.toDataURL("image/png", 1);
		return dataURL;
	}

}
