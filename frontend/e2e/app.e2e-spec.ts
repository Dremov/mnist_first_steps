import { RecogImagePage } from './app.po';

describe('recog-image App', () => {
  let page: RecogImagePage;

  beforeEach(() => {
    page = new RecogImagePage();
  });

  it('should display welcome message', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('Welcome to app!');
  });
});
