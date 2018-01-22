import { Component, ViewChild } from '@angular/core';
import { Http } from '@angular/http';
import { RequestOptions } from '@angular/http';
import { Observable} from 'rxjs/Observable';
import { Headers} from '@angular/http';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';
import { DrawComponent } from './draw/draw.component';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  headers: Headers;
  title = 'app';
  result: Observable<String>;
	imageLink: String;

	drawSelect = false;
	imageError = true;
	resultPresent: boolean = false;

	image: any;
	heatmap: any = undefined;
  res: any;
  accuracy: any;

	serverAddress = 'http://165.227.171.74:81';
	placeholder = 'http://www.euneighbours.eu/sites/default/files/2017-01/placeholder.png';

  // http://blog.otoro.net/assets/20160401/png/mnist_input_1.png

  constructor(private http: HttpClient, private sanitizer:DomSanitizer) {
		this.imageLink = this.placeholder;
		// this.getModelAccuracy();
	}

	@ViewChild(DrawComponent)
	private drawComponent: DrawComponent;
	getCanvasImg() {
		 let image = this.drawComponent.getSavedImage();
		 this.image = image;
		 this.recognizeNumber(image);
	}

  recognizeNumber(image) {
			let link = this.imageLink;
			let formData = new FormData();
			var imgBlob = new Blob([ image ], { type: "image/png" } );
			formData.append('image', imgBlob);

			let formHeaders = new HttpHeaders();
			formHeaders.set('Content-Type', 'multipart/form-data');

			this.http.post(this.serverAddress + '/image', formData, {
				headers: formHeaders })
      .subscribe(value => {
				console.log(value);
				this.res = value;
				let base64img = "data:image/png;base64," + this.res.heatmap;
				this.heatmap = this.sanitizer.bypassSecurityTrustResourceUrl(base64img);
				this.resultPresent = true;
			});
	}

	recognizeImage() {
		let link = this.imageLink;
		this.http.get(this.serverAddress + '/?url=' + link)
		.subscribe(value => {
			this.res = value;
			this.resultPresent = true;
		});
	}

  getModelAccuracy() {
    this.http.get(this.serverAddress + '/accuracy')
    .subscribe(value => {
			let acc: number = +value;
			this.accuracy = acc * 100;
		});
	}

	setPlaceholderImage() {
		this.imageLink = this.placeholder;
	}

  getImage($event) {
		this.imageLink = $event.target.value;
		this.imageError = false;
	}

	imageLinkError() {
		this.imageError = true;
		this.setPlaceholderImage();
	}



  upload(event) {
    // let fileList: FileList = event.target.files;
    // if(fileList.length > 0) {
    //     let file: File = fileList[0];
    //     let formData:FormData = new FormData();
    //     formData.append('uploadFile', file, file.name);
    //     let headers = new Headers();
    //     /** No need to include Content-Type in Angular 4 */
    //     headers.append('Content-Type', 'multipart/form-data');
    //     headers.append('Accept', 'application/json');
    //     let options = new RequestOptions({ headers: headers });
    //     this.http.post('api/endpoint', formData, options)
    //         .map(res => res.json())
    //         .catch(error => Observable.throw(error))
    //         .subscribe(
    //             data => console.log('success'),
    //             error => console.log(error)
    //         )
    // }
}
}
