import { Component } from '@angular/core';
import { Http } from '@angular/http';
import { RequestOptions } from '@angular/http';
import { Observable} from 'rxjs/Observable';
import { Headers} from '@angular/http';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  headers: Headers;
  title = 'app';
  file: any;
  result: Observable<String>;
  imageLink: String;
  res: any;

  // http://blog.otoro.net/assets/20160401/png/mnist_input_1.png

  constructor(private http: HttpClient) {
    this.imageLink = 'http://www.euneighbours.eu/sites/default/files/2017-01/placeholder.png';
  }

  recognizeImage() {
      const address = 'http://18.194.98.92'
      let link = this.imageLink;
      this.http.get('http://18.194.98.92/?url=' + link)
      .subscribe(value => this.res = value);
  }

  getImage($url) {
    this.imageLink = $url;
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