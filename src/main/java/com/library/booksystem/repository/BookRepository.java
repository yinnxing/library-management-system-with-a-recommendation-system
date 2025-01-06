package com.library.booksystem.repository;

import com.library.booksystem.model.Book;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;


@Repository
public interface BookRepository extends JpaRepository<Book, Integer>, JpaSpecificationExecutor<Book> {

}
