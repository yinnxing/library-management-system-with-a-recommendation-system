package com.library.booksystem.model.specification;

import com.library.booksystem.model.Transaction;
import com.library.booksystem.model.specification.criteria.TransactionCriteria;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.CriteriaQuery;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import lombok.RequiredArgsConstructor;
import org.springframework.data.jpa.domain.Specification;
import java.util.ArrayList;
import java.util.List;

@RequiredArgsConstructor

public class TransactionFilterSpecification implements Specification<Transaction> {
    private final TransactionCriteria criteria;

    @Override
    public Predicate toPredicate(Root<Transaction> root, CriteriaQuery<?> query, CriteriaBuilder builder) {
        if (criteria == null) {
            return null;
        }

        List<Predicate> predicates = new ArrayList<>();

        if (criteria.getUserId() != null) {
            predicates.add(builder.equal(root.get("user").get("userId"), criteria.getUserId()));
        }

        if (criteria.getBookId() != null) {
            predicates.add(builder.equal(root.get("book").get("bookId"), criteria.getBookId()));
        }

        if (criteria.getBorrowDateStart() != null) {
            predicates.add(builder.greaterThanOrEqualTo(root.get("borrowDate"), criteria.getBorrowDateStart()));
        }

        if (criteria.getBorrowDateEnd() != null) {
            predicates.add(builder.lessThanOrEqualTo(root.get("borrowDate"), criteria.getBorrowDateEnd()));
        }

        if (criteria.getDueDateStart() != null) {
            predicates.add(builder.greaterThanOrEqualTo(root.get("dueDate"), criteria.getDueDateStart()));
        }

        if (criteria.getDueDateEnd() != null) {
            predicates.add(builder.lessThanOrEqualTo(root.get("dueDate"), criteria.getDueDateEnd()));
        }

        if (criteria.getReturnDateStart() != null) {
            predicates.add(builder.greaterThanOrEqualTo(root.get("returnDate"), criteria.getReturnDateStart()));
        }

        if (criteria.getReturnDateEnd() != null) {
            predicates.add(builder.lessThanOrEqualTo(root.get("returnDate"), criteria.getReturnDateEnd()));
        }

        if (criteria.getStatus() != null) {
            predicates.add(builder.equal(root.get("status"), criteria.getStatus()));
        }

        if (criteria.getQ() != null) {
            String q = "%" + criteria.getQ().toLowerCase() + "%";
            predicates.add(
                    builder.or(
                            builder.like(builder.lower(root.get("user").get("userId")), q),
                            builder.like(builder.lower(root.get("book").get("bookId")), q)
                    )
            );
        }

        if (!predicates.isEmpty()) {
            query.where(predicates.toArray(new Predicate[0]));
        }

        return query.distinct(true).getRestriction();
    }
}
