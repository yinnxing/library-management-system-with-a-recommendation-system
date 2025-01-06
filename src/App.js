import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { publicRoutes, adminRoutes } from './routes/routes';  
import DefaultLayout from './layouts/userLayout';  
import { UserProvider } from './contexts/UserContext'; 
//import { jwtDecode } from 'jwt-decode'; 

function App() {
  const category = 'fiction';
  const isAdmin = true;
  // const isAdmin = () => {
  //   const token = localStorage.getItem('accessToken');
  //   if (!token) return false;

  //   try {
  //     const decodedToken = jwtDecode(token);
  //     return decodedToken.role === 'ADMIN'; 
  //   } catch (error) {
  //     return false;
  //   }
  // };  

  return (
  <UserProvider>
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

                {isAdmin && adminRoutes.map((route, index) => {
          const Page = route.component;

          return (
            <Route
              key={index}
              path={route.path}
              element={<Page />}
            >
              {/* Lặp qua các children (route con) của mỗi route */}
              {route.children && route.children.map((child, childIndex) => {
                const ChildPage = child.component;
                return (
                  <Route
                    key={childIndex}
                    path={child.path}
                    element={<ChildPage />}
                  />
                );
              })}
            </Route>
          );
        })}

      </Routes>
    </BrowserRouter>
  </UserProvider>
  );
}


export default App;
