import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { publicRoutes } from './routes/routes';  
import DefaultLayout from './layouts/userLayout';  
import { UserProvider } from './contexts/UserContext'; 
import { FavoritesProvider } from './contexts/FavouritesContext';  

function App() {
  const category = 'fiction';  

  return (
  <UserProvider>
    <FavoritesProvider>
    <BrowserRouter>
      <Routes>
        {publicRoutes.map((route, index) => {
          const Page = route.component;  
          const Layout = route.layout || DefaultLayout; 

          return (
            <Route
              key={index}
              path={route.path}
              element={
                <Layout>
                  <Page category={category} />
                </Layout>
              }
            >
              {/* Nếu có route con, bạn có thể render thêm ở đây */}
              {route.childPath && (
                <Route
                  path={route.childPath}
                  element={
                    <Layout>
                      <Page category={category} />
                    </Layout>
                  }
                />
              )}
            </Route>
          );
        })}
      </Routes>
    </BrowserRouter>
    </FavoritesProvider>
  </UserProvider>
  );
}

// export default App;
// import React from 'react';
// import BookDisplay from './pages/BookDisplay/BookDisplay';

// const bookData = {
//     title: "Effective Java",
//     author: "Joshua Bloch",
//     publisher: "Addison-Wesley",
//     publicationYear: 2018,
//     isbn: "9780134685991",
//     genre: "Programming",
//     description: "A comprehensive guide to best practices in Java programming.",
//     coverImageUrl: "https://covers.openlibrary.org/b/id/8231851-L.jpg",
//     previewLink: "http://books.google.com.vn/books?id=zf_bDwAAQBAJ&printsec=frontcover&dq=Python+programming&hl=&cd=4&source=gbs_api",
//     quantity: 30,
//     availableQuantity: 20,
// };
// /
// const App = () => (
//     <div>
//         <h1>Library Book Details</h1>
//         <BookDisplay book={bookData} />
//     </div>
// );


export default App;
